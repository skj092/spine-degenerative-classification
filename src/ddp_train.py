import os
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from transformers import get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from conf import get_config, transforms_train, transforms_val, device, set_seed

# Load configuration
config = get_config('local')

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


# ========================= Utility Functions ===========================

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # Use 'nccl' for GPUs; 'gloo' or 'mpi' for CPUs
        init_method="tcp://127.0.0.1:23456",  # Use a TCP address
        rank=rank,  # The rank of the current process
        world_size=world_size  # Total number of processes
    )
    torch.cuda.set_device(rank)  # Sets the device for this process

def cleanup():
    dist.destroy_process_group()

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def create_dataloader(df, phase, transform, batch_size, shuffle, drop_last, num_workers, sampler=None):
    dataset = RSNA24Dataset(df, phase=phase, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
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

    avg_loss = total_loss / len(train_dl)
    logger.info(f'Training Loss: {avg_loss:.6f}')
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
    return avg_loss, y_preds, labels

def train_and_validate(df, n_folds, seed, model_params, train_params, rank, output_dir, logger, callbacks=None):
    if callbacks is None:
        callbacks = []

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        if rank == 0:
            logger.info(f'\n{"#"*30}\nStart fold {fold}\n{"#"*30}\n')
        print(f"Fold: {fold}")
        if rank == 0:
            logger.info(
                f'{len(trn_idx)} training samples, {len(val_idx)} validation samples')

        df_train, df_valid = df.iloc[trn_idx], df.iloc[val_idx]
        train_sampler = torch.utils.data.distributed.DistributedSampler(df_train, num_replicas=dist.get_world_size(), rank=rank)
        train_dl = create_dataloader(
            df_train, 'train', train_params['transform_train'], train_params['batch_size'], True, True, train_params['n_workers'], sampler=train_sampler)
        valid_dl = create_dataloader(
            df_valid, 'valid', train_params['transform_val'], train_params['batch_size'] * 2, False, False, train_params['n_workers'])

        model, optimizer = create_model_and_optimizer(
            **model_params, device=rank)

        # Wrap the model with DistributedDataParallel
        model = DDP(model, device_ids=[rank])

        scaler = GradScaler(enabled=config.USE_AMP, init_scale=4096)
        scheduler = create_scheduler(optimizer, len(
            train_dl), train_params['epochs'], train_params['grad_acc'])

        checkpoint_path = f"{output_dir}/checkpoint_fold_{fold}.pth"
        start_epoch = 0
        best_loss = float('inf')
        if os.path.exists(checkpoint_path):
            start_epoch, best_loss = load_checkpoint(
                model, optimizer, scheduler, scaler, checkpoint_path, rank)
            if rank == 0:
                logger.info(
                    f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")

        for callback in callbacks:
            callback.on_train_begin()

        for epoch in range(start_epoch, train_params['epochs']):
            if rank == 0:
                logger.info(f'Start epoch {epoch}')
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            train_loss = train_one_epoch(
                model, train_dl, train_params['criterion'], optimizer, scheduler, scaler, train_params['grad_acc'], rank, logger)
            if rank == 0:
                logger.info(f'train_loss: {train_loss:.6f}')

            val_loss, y_preds, labels = validate_one_epoch(
                model, valid_dl, train_params['criterion'], rank, logger)
            val_wll = train_params['criterion2'](y_preds, labels)
            if rank == 0:
                logger.info(f'val_loss: {val_loss:.6f}, val_wll: {val_wll:.6f}')

            logs = {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'scaler': scaler,
                'val_loss': val_loss,
                'val_wll': val_wll,
                'fold': fold
            }
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            if val_loss < best_loss:
                best_loss = val_loss
                if rank == 0:
                    logger.info(f'New best loss: {best_loss:.6f}')
                save_checkpoint(model, optimizer, scheduler,
                                scaler, epoch, fold, best_loss, output_dir)

            if any(callback.early_stop for callback in callbacks if isinstance(callback, EarlyStopping)):
                if rank == 0:
                    logger.info('Early stopping')
                break

        # Ensure the best model is saved at the end of the fold, even if early stopping occurs
        if rank == 0:
            logger.info(
                f"Saving the best model for fold {fold} with loss {best_loss:.6f}")
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, fold, best_loss, output_dir)

        for callback in callbacks:
            callback.on_train_end()


# ========================= Evaluation Functions ============================

def evaluate_model(df, skf, model_class, model_params, transforms_val, device, output_dir, n_workers, n_labels, logger):
    all_y_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device))

    for fold, (_, val_idx) in enumerate(skf.split(range(len(df)))):
        logger.info(f'\n{"#"*30}\nStart fold {fold}\n{"#"*30}\n')
        print(f"Fold: {fold}")

        df_valid = df.iloc[val_idx]
        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=n_workers)

        model = model_class(**model_params)
        checkpoint_path = f'{output_dir}/best_model_fold_{fold}.pth'

        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping this fold.")
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

    if all_y_preds:
        y_preds = torch.cat(all_y_preds)
        labels = torch.cat(all_labels)
        logger.info(f'Evaluation complete for {len(all_y_preds)} folds.')
        return y_preds, labels
    else:
        logger.error("No valid folds found for evaluation.")
        return None, None


def compute_cv_score(y_preds, labels, weights, logger):
    # Ensure weights is a tensor before calling .cpu()
    if isinstance(weights, list):
        weights = torch.tensor(weights)
    weights_np = weights.cpu().numpy() if weights is not None else None
    # Convert y_preds to float32 and then to numpy array
    y_preds_np = y_preds.to(torch.float32).softmax(1).cpu().numpy()

    # Ensure labels and weights are also numpy arrays
    labels_np = labels.cpu().numpy()

    cv = log_loss(labels_np, y_preds_np,
                  normalize=True, sample_weight=weights_np)
    logger.info(f'CV Score: {cv:.6f}')
    return cv


def compute_random_score(labels, n_classes, weights, logger):
    random_pred = np.ones((labels.shape[0], n_classes)) / n_classes
    random_score = log_loss(labels, random_pred,
                            normalize=True, sample_weight=weights)
    logger.info(f'Random Score: {random_score:.6f}')
    return random_score


def save_predictions_and_labels(y_preds, labels, output_dir, logger):
    np.save(f'{output_dir}/labels.npy', labels.numpy())
    np.save(f'{output_dir}/final_oof.npy', y_preds.float().numpy())
    logger.info(f'Predictions and labels saved to {output_dir}')


# ========================= Dataset Class ============================

class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((512, 512, config.IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)

        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                p = f'./cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                pass

        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'./cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                pass

        # Axial T2
        axt2 = glob(f'./cvt_png/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)

        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2)+0.0001

        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                pass

        assert np.sum(x) > 0

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)

        return x, label


# ========================= Main Script ============================

def main(data_dir: str):
    # Manually set environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0'  # Set this to '1' for the second GPU

    # Manually set rank and world_size for Kaggle
    world_size = 2  # Number of GPUs available
    rank = int(os.environ['RANK'])  # Adjust based on the GPU ID

    # Initialize process group and set up GPU device
    setup(rank, world_size)

    # Initialize directories and seed
    set_seed(config.SEED)
    output_dir = f'rsna24-results'
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    if rank == 0:  # Log only for the master process
        log_file = os.path.join(output_dir, 'training_log.txt')
        logger = setup_logger(log_file)
    else:
        logger = None

    # Data Preprocessing
    df = pd.read_csv(f"{data_dir}/train.csv")
    subset_size = config.subset_size
    if subset_size:
        df = df.sample(n=subset_size, random_state=config.SEED)
    if logger:
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

    # Initialize callbacks only on the master process
    if rank == 0:
        callbacks = [
            ModelCheckpoint(output_dir=output_dir, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=5, mode='min')
        ]
    else:
        callbacks = []

    # Train and validate the model
    train_and_validate(df, config.N_FOLDS, config.SEED, model_params,
                       train_params, rank, output_dir, logger, callbacks=callbacks)

    # Clean up the process group
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    main(parser.parse_args().data_dir)

