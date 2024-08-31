from collections import OrderedDict
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from config import MODEL_NAME
from config import USE_AMP
from models import RSNA24Model
from torch.optim import AdamW
from config import EPOCHS
from config import transforms_train, transforms_val, IN_CHANS, N_CLASSES, BATCH_SIZE, GRAD_ACC, SEED, N_FOLDS, LR, WD, device
from sklearn.model_selection import KFold
import torch
import os
import pandas as pd
from pathlib import Path
from dataset import RSNA24Dataset
from torch.utils.data import DataLoader
from config import set_seed
from transformers import get_cosine_schedule_with_warmup


# =========================Training===========================
def create_dataloader(df, phase, transform, batch_size, shuffle, drop_last, num_workers):
    dataset = RSNA24Dataset(df, phase=phase, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers
    )


def create_model_and_optimizer(model_name, in_chans, n_classes, lr, wd, device):
    model = RSNA24Model(model_name, in_chans, n_classes, pretrained=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return model, optimizer


def create_scheduler(optimizer, train_loader_len, epochs, grad_acc):
    warmup_steps = epochs / 10 * train_loader_len // grad_acc
    num_total_steps = epochs * train_loader_len // grad_acc
    num_cycles = 0.475
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_total_steps, num_cycles=num_cycles)


def train_one_epoch(model, train_dl, criterion, optimizer, scheduler, scaler, grad_acc, device):
    model.train()
    total_loss = 0
    with tqdm(train_dl, leave=True) as pbar:
        optimizer.zero_grad()
        for idx, (x, t) in enumerate(pbar):
            x, t = x.to(device), t.to(device)
            with autocast(enabled=USE_AMP, dtype=torch.half):
                loss = 0
                y = model(x)
                for col in range(N_LABELS):
                    pred = y[:, col*3:col*3+3]
                    gt = t[:, col]
                    loss += criterion(pred, gt) / N_LABELS

                total_loss += loss.item()
                if grad_acc > 1:
                    loss /= grad_acc

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM or 1e9)
            if (idx + 1) % grad_acc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            pbar.set_postfix(OrderedDict(
                loss=f'{loss.item()*grad_acc:.6f}', lr=f'{optimizer.param_groups[0]["lr"]:.3e}'))

    return total_loss / len(train_dl)


def validate_one_epoch(model, valid_dl, criterion, device):
    model.eval()
    total_loss = 0
    y_preds, labels = [], []

    with torch.no_grad():
        with tqdm(valid_dl, leave=True) as pbar:
            for x, t in pbar:
                x, t = x.to(device), t.to(device)
                with autocast(enabled=USE_AMP, dtype=torch.half):
                    loss = 0
                    y = model(x)
                    for col in range(N_LABELS):
                        pred = y[:, col*3:col*3+3]
                        gt = t[:, col]
                        loss += criterion(pred, gt) / N_LABELS
                        y_preds.append(pred.cpu())
                        labels.append(gt.cpu())
                    total_loss += loss.item()

    return total_loss / len(valid_dl), torch.cat(y_preds), torch.cat(labels)


def save_best_model(model, val_loss, val_wll, best_loss, best_wll, fold, epoch, device, output_dir):
    if val_loss < best_loss or val_wll < best_wll:
        if device != 'cuda:0':
            model.to('cuda:0')

        if val_loss < best_loss:
            print(
                f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
            best_loss = val_loss

        if val_wll < best_wll:
            print(
                f'epoch:{epoch}, best wll_metric updated from {best_wll:.6f} to {val_wll:.6f}')
            best_wll = val_wll
            fname = f'{output_dir}/best_wll_model_fold-{fold}.pt'
            torch.save(model.state_dict(), fname)

        if device != 'cuda:0':
            model.to(device)

        return best_loss, best_wll, 0
    return best_loss, best_wll, 1


def train_and_validate(df, n_folds, seed, model_params, train_params, device, output_dir):
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_loss, best_wll, es_step = 1.2, 1.2, 0

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print(f'\n{"#"*30}\nstart fold {fold}\n{"#"*30}\n')
        print(f'{len(trn_idx)} training samples, {len(val_idx)} validation samples')

        df_train, df_valid = df.iloc[trn_idx], df.iloc[val_idx]
        train_dl = create_dataloader(
            df_train, 'train', train_params['transform_train'], train_params['batch_size'], True, True, train_params['n_workers'])
        valid_dl = create_dataloader(
            df_valid, 'valid', train_params['transform_val'], train_params['batch_size']*2, False, False, train_params['n_workers'])

        model, optimizer = create_model_and_optimizer(
            **model_params, device=device)
        scaler = GradScaler(enabled=USE_AMP, init_scale=4096)
        scheduler = create_scheduler(optimizer, len(
            train_dl), train_params['epochs'], train_params['grad_acc'])

        for epoch in range(1, train_params['epochs'] + 1):
            print(f'start epoch {epoch}')
            train_loss = train_one_epoch(
                model, train_dl, train_params['criterion'], optimizer, scheduler, scaler, train_params['grad_acc'], device)
            print(f'train_loss: {train_loss:.6f}')

            val_loss, y_preds, labels = validate_one_epoch(
                model, valid_dl, train_params['criterion'], device)
            val_wll = train_params['criterion2'](y_preds, labels)
            print(f'val_loss: {val_loss:.6f}, val_wll: {val_wll:.6f}')

            best_loss, best_wll, es_step = save_best_model(
                model, val_loss, val_wll, best_loss, best_wll, fold, epoch, device, output_dir)
            if es_step >= train_params['early_stopping_epoch']:
                print('early stopping')
                break


# Example of usage:
N_WORKERS = os.cpu_count()
OUTPUT_DIR = f'rsna24-results'
SEED = 862
EARLY_STOPPING_EPOCH = 3
set_seed(42)
rd = Path(__file__).parent.parent

# Data Preprocessing
df = pd.read_csv(rd / 'data/train.csv')
df = df.sample(n=100, random_state=SEED)
print(df.head())

# fill missing values
df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

CONDITIONS = [
    'Spinal Canal Stenosis',
    'Left Neural Foraminal Narrowing',
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]


MAX_GRAD_NORM = None
N_LABELS = 25


model_params = {
    'model_name': MODEL_NAME,
    'in_chans': IN_CHANS,
    'n_classes': N_CLASSES,
    'lr': LR,
    'wd': WD
}

train_params = {
    'transform_train': transforms_train,
    'transform_val': transforms_val,
    'batch_size': BATCH_SIZE,
    'n_workers': N_WORKERS,
    'epochs': EPOCHS,
    'grad_acc': GRAD_ACC,
    'criterion': nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]).to(device)),
    'criterion2': nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0])),
    'early_stopping_epoch': EARLY_STOPPING_EPOCH
}

train_and_validate(df, N_FOLDS, SEED, model_params,
                   train_params, device, OUTPUT_DIR)
