import argparse
from models import EarlyStopping, ModelCheckpoint
from engine import train_and_validate, evaluate_model
import wandb
from models import RSNA24Model
from conf import (
    transforms_train, transforms_val,
    device, set_seed
)
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from conf import get_config
from utils import (setup_logger, compute_cv_score,
                   save_predictions_and_labels)

def main(data_dir: str, config: dict):
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
                       train_params, device, output_dir, logger, callbacks=callbacks, config=config)

    # Evaluate the model
    skf = KFold(n_splits=config.N_FOLDS, shuffle=True,
                random_state=config.SEED)
    y_preds, labels = evaluate_model(
        df, skf, RSNA24Model, model_params, transforms_val, device, output_dir, os.cpu_count(), config.N_LABELS, logger, config=config)

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
    os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
    wandb.init(project="rsna24", config=config)
    print(f"Using config: {config}")
    print(f"Data directory: {parser.parse_args().data_dir}")
    main(data_dir=parser.parse_args().data_dir, config=config)
