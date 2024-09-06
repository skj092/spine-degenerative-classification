import torch
from torch.optim import AdamW
from torch import nn
from transformers import get_cosine_schedule_with_warmup
import timm
import os


def load_weights_skip_mismatch(model, weights_path, device):
    # Load Weights
    state_dict = torch.load(weights_path, map_location=device)
    model_dict = model.state_dict()

    # Iter models
    params = {}
    for (sdk, sfv), (mdk, mdv) in zip(state_dict.items(), model_dict.items()):
        if sfv.size() == mdv.size():
            params[sdk] = sfv
        else:
            print("Skipping param: {}, {} != {}".format(
                sdk, sfv.size(), mdv.size()))

    # Reload + Skip
    model.load_state_dict(params, strict=False)
    print("Loaded weights from:", weights_path)


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_chans=10, n_classes=75, pretrained=True, features_only=False, lr=2e-4, wd=1e-2, config=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans,
                                       num_classes=n_classes, features_only=features_only, global_pool='avg')
        if pretrained:
            f = "{}_{}.pt".format(config.backbone, config.seed)
            load_weights_skip_mismatch(self.model, f, config.device)
            print("Loaded weights from:", f)

    def forward(self, xb):
        return self.model(xb)


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


def create_model_and_optimizer(model_name, in_chans, n_classes, lr, wd, device, config=None):
    model = RSNA24Model(model_name=model_name,
                        in_chans=in_chans, n_classes=n_classes, config=config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return model, optimizer


def create_scheduler(optimizer, train_loader_len, epochs, grad_acc):
    warmup_steps = epochs / 10 * train_loader_len // grad_acc
    num_total_steps = epochs * train_loader_len // grad_acc
    num_cycles = 0.475
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_total_steps, num_cycles=num_cycles)


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
