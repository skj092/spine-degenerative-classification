# Pytorch Lightning training script for the RSNA24 dataset
from transformers import get_cosine_schedule_with_warmup
from torch import nn
import json
import argparse
from sklearn.model_selection import KFold
import pandas as pd
from dataset import RSNA24Dataset
from models import RSNA24Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from conf import get_transforms
import wandb
from torch.cuda.amp import autocast


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# DataModule Class
class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, config):
        super().__init__()
        self.config = config
        self.train_transform, self.val_transform = get_transforms(config)
        self.train_df = train_df
        self.valid_df = valid_df

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if self.config.DEBUG:
            self.train_df = self.train_df.sample(4)
        train_dataset = RSNA24Dataset(
            df=self.train_df, config=self.config, transform=self.train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        if self.config.DEBUG:
            self.valid_df = self.valid_df.sample(4)
        val_dataset = RSNA24Dataset(
            df=self.valid_df, config=self.config, transform=self.val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4)
        return val_loader


# Lightning Module class


class LightningModel(pl.LightningModule):
    def __init__(self, model, config, grad_accum=1):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0, 4.0]))
        self.grad_accum = grad_accum

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)
        total_loss = 0

        with autocast(enabled=config.USE_AMP, dtype=torch.half):
            y_hat = self.model(x)  # (bs, 75)
            loss = 0
            for col in range(self.config.N_LABELS):
                pred = y_hat[:, col*3:col*3+3]  # (bs, 3)
                gt = y[:, col]  # (bs)
                loss += self.loss_fn(pred, gt) / self.config.N_CLASSES
        total_loss += loss
        # log average training loss for all the training dataloader

        # Log training loss and learning rate
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)
        with autocast(enabled=config.USE_AMP, dtype=torch.half):
            y_hat = self.model(x)
            loss = 0
            for col in range(self.config.N_LABELS):
                pred = y_hat[:, col*3:col*3+3]
                gt = y[:, col]
                loss = self.loss_fn(pred, gt) / self.config.N_CLASSES

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def create_scheduler(optimizer, train_loader_len, epochs, grad_acc):
        warmup_steps = epochs / 10 * train_loader_len // grad_acc
        num_total_steps = epochs * train_loader_len // grad_acc
        num_cycles = 0.475
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_total_steps, num_cycles=num_cycles)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.LR)
        return [optimizer]


# Main function


def main(config: dict):
    # data preprocessing
    df = pd.read_csv(f'{config.CSV_PATH}/train.csv')
    df.fillna(-100, inplace=True)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.map(lambda x: label2id.get(x, x))

    # KFold Cross Validation
    skf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    # If wandb logging is enabled then initialize wandb else use CSVLogger
    if config.USE_WANDB:
        wandb.login(key=config.WANDB_API_KEY)
        wandb.init(project='RSNA24')
    else:
        logger = CSVLogger('logs/', name='RSNA24')

    # Train the model
    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):

        # prepare dataloader
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        data_module = DataModule(train_df, val_df, config)
        data_module.setup()

        # Model
        model = RSNA24Model(model_name=config.MODEL_NAME, in_chans=30)
        model = LightningModel(model, config)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        )
        trainer = pl.Trainer(
            accelerator=config.device,
            max_epochs=config.EPOCHS,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback]
        )
        trainer.fit(model, data_module)
    if config.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    # Take config from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    config_dict = json.load(open(parser.parse_args().cfg))
    config = Config(config_dict)
    print(f"Using config: {config}")
    main(config)
