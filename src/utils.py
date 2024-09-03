import wandb
import torch
import numpy as np
import logging
from sklearn.metrics import log_loss


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


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
