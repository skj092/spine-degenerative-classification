from dataset import create_dataloader
from models import create_model_and_optimizer, save_checkpoint, create_scheduler
import wandb
from dataset import RSNA24Dataset
import os
import torch
from tqdm import tqdm
from collections import OrderedDict
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import KFold


def train_one_epoch(model, train_dl, criterion, optimizer, scheduler, scaler, grad_acc, device, logger, config=None):
    model.train()
    total_loss = 0
    with tqdm(train_dl, leave=True) as pbar:
        optimizer.zero_grad()
        for idx, (x, t) in enumerate(pbar):
            x, t = x.to(device), t.to(device)
            with autocast(device_type=str(device), enabled=config.USE_AMP, dtype=torch.half):
                loss = 0
                y = model(x)
                # calculating the loss separately for each label or class ensures that each output is given the appropriate attention.
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
            if config.USE_WANDB:
                wandb.log({"train_loss": loss.item() * grad_acc,
                      "learning_rate": optimizer.param_groups[0]["lr"]})

    avg_loss = total_loss / len(train_dl)
    logger.info(f'Training Loss: {avg_loss:.6f}')
    if config.USE_WANDB:
        wandb.log({"avg_train_loss": avg_loss})
    return avg_loss


def validate_one_epoch(model, valid_dl, criterion, device, logger, config=None):
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
    if config.USE_WANDB:
        wandb.log({"val_loss": avg_loss})
    return avg_loss, y_preds, labels


def train_and_validate(df, n_folds, seed, model_params, train_params, device, output_dir, logger, callbacks=None, config=None):
    if callbacks is None:
        callbacks = []

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_loss, best_wll, es_step = 1.2, 1.2, 0

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        logger.info(f'\n{"#"*30}\nStart fold {fold}\n{"#"*30}\n')
        if config.USE_WANDB:
            wandb.log({"fold": fold})
        logger.info(
            f'{len(trn_idx)} training samples, {len(val_idx)} validation samples')

        df_train, df_valid = df.iloc[trn_idx], df.iloc[val_idx]
        train_dl = create_dataloader(
            df_train, 'train', train_params['transform_val'], train_params['batch_size'], True, True, train_params['n_workers'], config=config)
        valid_dl = create_dataloader(
            df_valid, 'valid', train_params['transform_val'], train_params['batch_size']*2, False, False, train_params['n_workers'], config=config)

        model, optimizer = create_model_and_optimizer(
            **model_params, device=device)
        scaler = GradScaler(enabled=config.USE_AMP, init_scale=4096)
        scheduler = create_scheduler(optimizer, len(
            train_dl), train_params['epochs'], train_params['grad_acc'])

        for callback in callbacks:
            callback.on_train_begin()

        for epoch in range(train_params['epochs']):
            logger.info(f'Start epoch {epoch}')
            if config.USE_WANDB:
                wandb.log({"epoch": epoch})

            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            train_loss = train_one_epoch(
                model, train_dl, train_params['criterion'], optimizer, scheduler, scaler, train_params['grad_acc'], device, logger, config=config)
            logger.info(f'train_loss: {train_loss:.6f}')

            val_loss, y_preds, labels = validate_one_epoch(
                model, valid_dl, train_params['criterion'], device, logger, config=config)
            val_wll = train_params['criterion2'](y_preds, labels)
            logger.info(f'val_loss: {val_loss:.6f}, val_wll: {val_wll:.6f}')
            if config.USE_WANDB:
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


def evaluate_model(df, skf, model_class, model_params, transforms_val, device, output_dir, n_workers, n_labels, logger, config=None):
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
        # torch.save(checkpoint, f"{output_dir}/checkpoint_fold_{fold}.pth")
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
