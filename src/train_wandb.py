import argparse
import wandb
from dataset import RSNA24Dataset
from models import RSNA24Model
from conf import (
    transforms_train, transforms_val,
    device, set_seed
)
import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from transformers import get_cosine_schedule_with_warmup
from conf import get_config


# ========================= Callback Classes ===========================

class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, mode='min'):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.best is None or (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping at epoch {epoch}")

# ========================= Utility Functions ===========================


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def create_dataloader(df, phase, transform, batch_size, shuffle, drop_last, num_workers):
    dataset = RSNA24Dataset(df, phase=phase, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=2
    )


def create_model_and_optimizer(model_name, in_chans, n_classes, lr, wd, device):
    model = RSNA24Model(model_name=model_name,
                        in_chans=in_chans, n_classes=n_classes)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return model, optimizer


def create_scheduler(optimizer, train_loader_len, epochs, grad_acc):
    warmup_steps = epochs / 10 * train_loader_len // grad_acc
    num_total_steps = epochs * train_loader_len // grad_acc
    num_cycles = 0.475
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_total_steps, num_cycles=num_cycles)


def train_one_epoch(model, train_dl, criterion, optimizer, scheduler, scaler, grad_acc, device, logger):
    model.train()
    total_loss = 0
    with tqdm(train_dl, leave=True) as pbar:
        optimizer.zero_grad()
        for idx, (x, t) in enumerate(pbar):
            x, t = x.to(device), t.to(device)
            with autocast(device_type=str(device), enabled=config.USE_AMP, dtype=torch.half):
                loss = 0
                y = model(x)
                for col in range(config.N_LABELS):
                    pred = y[:, col*3:col*3+3]
                    gt = t[:, col]
                    loss += criterion(pred, gt) / config.N_LABELS

                total_loss += loss.item()
                if grad_acc > 1:
                    loss /= grad_acc

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.MAX_GRAD_NORM or 1e9)
            if (idx + 1) % grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            pbar.set_postfix(OrderedDict(
                loss=f'{loss.item()*grad_acc:.6f}', lr=f'{optimizer.param_groups[0]["lr"]:.3e}'))
            wandb.log({"train_loss": loss.item() * grad_acc,
                      "learning_rate": optimizer.param_groups[0]["lr"]})

    avg_loss = total_loss / len(train_dl)
    logger.info(f'Training Loss: {avg_loss:.6f}')
    wandb.log({"avg_train_loss": avg_loss})
    return avg_loss


def validate_one_epoch(model, valid_dl, criterion, device, logger):
    model.eval()
    total_loss = 0
    y_preds, labels = [], []

    with torch.no_grad():
        with tqdm(valid_dl, leave=True) as pbar:
            for x, t in pbar:
                x, t = x.to(device), t.to(device)
                with autocast(device_type=str(device), enabled=config.USE_AMP, dtype=torch.half):
                    y = model(x)

                loss = 0
                for col in range(config.N_LABELS):
                    pred = y[:, col*3:col*3+3].float()
                    gt = t[:, col].long()
                    loss += criterion(pred, gt) / config.N_LABELS

                    y_preds.append(pred.cpu())
                    labels.append(gt.cpu())

                total_loss += loss.item()

    avg_loss = total_loss / len(valid_dl)
    y_preds = torch.cat(y_preds)
    labels = torch.cat(labels)

    logger.info(f'Validation Loss: {avg_loss:.6f}')
    wandb.log({"val_loss": avg_loss})
    return avg_loss, y_preds, labels


class ModelCheckpoint(Callback):
    def __init__(self, output_dir, monitor='val_loss', mode='min'):
        self.output_dir = output_dir
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.best is None or (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            self.best_epoch = epoch
            model_to_save = logs['model'].module if hasattr(
                logs['model'], 'module') else logs['model']

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': logs['optimizer'].state_dict(),
                'loss': current,
            }
            if 'scheduler' in logs:
                checkpoint['scheduler_state_dict'] = logs['scheduler'].state_dict()
            if 'scaler' in logs:
                checkpoint['scaler_state_dict'] = logs['scaler'].state_dict()

            torch.save(
                checkpoint, f"{self.output_dir}/best_model_fold_{logs['fold']}.pth")
            print(
                f"Saved best model at epoch {epoch} with {self.monitor}: {current:.4f}")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, fold, loss, output_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint_fold_{fold}.pth")


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


# ========================= Main Training and Validation Function ============================

def train_and_validate(df, n_folds, seed, model_params, train_params, device, output_dir, logger, callbacks=None):
    if callbacks is None:
        callbacks = []

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_loss, best_wll, es_step = 1.2, 1.2, 0

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        logger.info(f'\n{"#"*30}\nStart fold {fold}\n{"#"*30}\n')
        wandb.log({"fold": fold})
        logger.info(
            f'{len(trn_idx)} training samples, {len(val_idx)} validation samples')

        df_train, df_valid = df.iloc[trn_idx], df.iloc[val_idx]
        train_dl = create_dataloader(
            df_train, 'train', train_params['transform_train'], train_params['batch_size'], True, True, train_params['n_workers'])
        valid_dl = create_dataloader(
            df_valid, 'valid', train_params['transform_val'], train_params['batch_size']*2, False, False, train_params['n_workers'])

        model, optimizer = create_model_and_optimizer(
            **model_params, device=device)
        scaler = GradScaler(enabled=config.USE_AMP, init_scale=4096)
        scheduler = create_scheduler(optimizer, len(
            train_dl), train_params['epochs'], train_params['grad_acc'])

        for callback in callbacks:
            callback.on_train_begin()

        for epoch in range(train_params['epochs']):
            logger.info(f'Start epoch {epoch}')
            wandb.log({"epoch": epoch})

            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            train_loss = train_one_epoch(
                model, train_dl, train_params['criterion'], optimizer, scheduler, scaler, train_params['grad_acc'], device, logger)
            logger.info(f'train_loss: {train_loss:.6f}')

            val_loss, y_preds, labels = validate_one_epoch(
                model, valid_dl, train_params['criterion'], device, logger)
            val_wll = train_params['criterion2'](y_preds, labels)
            logger.info(f'val_loss: {val_loss:.6f}, val_wll: {val_wll:.6f}')
            wandb.log({"val_loss": val_loss, "val_wll": val_wll})

            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, fold, val_loss, output_dir)

            if es_step >= train_params['early_stopping_epoch']:
                logger.info('Early stopping')
                break

        for callback in callbacks:
            callback.on_train_end()

# ========================= Evaluation Functions ============================


def evaluate_model(df, skf, model_class, model_params, transforms_val, device, output_dir, n_workers, n_labels, logger):
    all_y_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0, 4.0]).to(device))

    for fold, (_, val_idx) in enumerate(skf.split(range(len(df)))):
        logger.info(f'\n{"#"*30}\nStart fold {fold}\n{"#"*30}\n')
        wandb.log({"fold_evaluation": fold})
        print(f"Fold: {fold}")

        df_valid = df.iloc[val_idx]
        valid_ds = RSNA24Dataset(
            df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, prefetch_factor=2,
                              pin_memory=True, drop_last=False, num_workers=n_workers)

        model = model_class(**model_params)
        # Load the checkpoint of the last fold
        #torch.save(checkpoint, f"{output_dir}/checkpoint_fold_{fold}.pth")
        checkpoint_path = f"{output_dir}/checkpoint_fold_{fold}.pth"


        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping this fold.")
            wandb.log(
                {"warning": f"Checkpoint for fold {fold} not found. Skipping this fold."})
            continue

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        fold_y_preds = []
        fold_labels = []

        with torch.no_grad():
            with tqdm(valid_dl, leave=True) as pbar:
                for x, t in pbar:
                    x, t = x.to(device), t.to(device)
                    with autocast(enabled=config.USE_AMP, device_type=str(device)):
                        y = model(x)
                        for col in range(n_labels):
                            pred = y[:, col*3:col*3+3]
                            gt = t[:, col]
                            fold_y_preds.append(pred.cpu())
                            fold_labels.append(gt.cpu())

        if fold_y_preds:
            all_y_preds.append(torch.cat(fold_y_preds))
            all_labels.append(torch.cat(fold_labels))

    y_preds = torch.cat(all_y_preds)
    labels = torch.cat(all_labels)
    logger.info(f'Evaluation complete for {len(all_y_preds)} folds.')
    wandb.log({"evaluation_complete": True})
    return y_preds, labels


def compute_cv_score(y_preds, labels, weights, logger):
    if isinstance(weights, list):
        weights = torch.tensor(weights)
    weights_np = weights.cpu().numpy() if weights is not None else None
    y_preds_np = y_preds.to(torch.float32).softmax(1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    cv = log_loss(labels_np, y_preds_np,
                  normalize=True, sample_weight=weights_np)
    logger.info(f'CV Score: {cv:.6f}')
    wandb.log({"cv_score": cv})
    return cv


def compute_random_score(labels, n_classes, weights, logger):
    random_pred = np.ones((labels.shape[0], n_classes)) / n_classes
    random_score = log_loss(labels, random_pred,
                            normalize=True, sample_weight=weights)
    logger.info(f'Random Score: {random_score:.6f}')
    wandb.log({"random_score": random_score})
    return random_score


def save_predictions_and_labels(y_preds, labels, output_dir, logger):
    np.save(f'{output_dir}/labels.npy', labels.numpy())
    np.save(f'{output_dir}/final_oof.npy', y_preds.float().numpy())
    logger.info(f'Predictions and labels saved to {output_dir}')
    wandb.save(f'{output_dir}/labels.npy')
    wandb.save(f'{output_dir}/final_oof.npy')

# ========================= Main Script ============================


def main(data_dir: str):
    # Initialize directories and seed
    set_seed(config.SEED)
    output_dir = 'rsna24-results'
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, 'training_log.txt')
    logger = setup_logger(log_file)
    wandb.run.name = "RSNA24 Training"
    wandb.run.save()

    # Data Preprocessing
    df = pd.read_csv(f'{data_dir}/train.csv')
    subset_size = config.subset_size
    if subset_size:
        print(f"Using subset of size: {subset_size}")
        df = df.sample(n=subset_size, random_state=config.SEED)
    logger.info(f"DataFrame shape: {df.shape}")
    df.fillna(-100, inplace=True)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.map(lambda x: label2id.get(x, x))

    model_params = {
        'model_name': config.MODEL_NAME,
        'in_chans': config.IN_CHANS,
        'n_classes': config.N_CLASSES,
        'lr': config.LR,
        'wd': config.WD,
    }

    train_params = {
        "lr": config.LR,
        "wd": config.WD,
        'transform_train': transforms_train,
        'transform_val': transforms_val,
        'batch_size': config.BATCH_SIZE,
        'n_workers': os.cpu_count(),
        'epochs': config.EPOCHS,
        'grad_acc': config.GRAD_ACC,
        'criterion': nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device)),
        'criterion2': nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0])),
        'early_stopping_epoch': config.EARLY_STOPPING_EPOCH
    }

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(output_dir=output_dir, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]

    # Train and validate the model
    train_and_validate(df, config.N_FOLDS, config.SEED, model_params,
                       train_params, device, output_dir, logger, callbacks=callbacks)

    # Evaluate the model
    skf = KFold(n_splits=config.N_FOLDS, shuffle=True,
                random_state=config.SEED)
    y_preds, labels = evaluate_model(
        df, skf, RSNA24Model, model_params, transforms_val, device, output_dir, os.cpu_count(), config.N_LABELS, logger)

    # Compute CV score and random score
    weights = [1 if l == 0 else 2 if l == 1 else 4 for l in labels.numpy()]
    cv_score = compute_cv_score(y_preds, labels, weights, logger)
    print(f"CV Score: {cv_score:.6f}")

    random_pred = np.ones((y_preds.shape[0], 3)) / 3.0
    random_score = log_loss(labels, random_pred,
                            normalize=True, sample_weight=weights)
    print(f"Random Score: {random_score:.6f}")

    # Save predictions and labels
    save_predictions_and_labels(y_preds, labels, output_dir, logger)


if __name__ == "__main__":
    # Take config from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="local")
    parser.add_argument("--data_dir", type=str, default="data")
    config = get_config(parser.parse_args().env)
    wandb.init(project="rsna24", config=config)
    print(f"Using config: {config}")
    print(f"Data directory: {parser.parse_args().data_dir}")
    main(data_dir=parser.parse_args().data_dir)
