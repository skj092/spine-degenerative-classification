import albumentations as A
import torch
import os
import random
import numpy as np


class Config:
    NOT_DEBUG = True
    OUTPUT_DIR = 'rsna24-results'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_WORKERS = os.cpu_count()
    USE_AMP = True
    SEED = 8620
    IMG_SIZE = [512, 512]
    IN_CHANS = 30
    N_LABELS = 25
    N_CLASSES = 3 * N_LABELS
    AUG_PROB = 0.75
    GRAD_ACC = 2
    MAX_GRAD_NORM = None
    EARLY_STOPPING_EPOCH = 3
    LR = 2e-4 * 64 / 32
    WD = 1e-2
    AUG = True
    N_FOLDS = 5
    EPOCHS = 20
    MODEL_NAME = "tf_efficientnet_b3.ns_jft_in1k"
    BATCH_SIZE = 32 // GRAD_ACC
    subset_size = None
    WWANDB_API_KEY="97b5307e24cc3a77259ade3057e4eea6fd2addb0"


class LocalConfig(Config):
    # Override or extend settings specific to local running
    BATCH_SIZE = 1
    N_WORKERS = 0
    NOT_DEBUG = True
    MODEL_NAME = "tf_efficientnet_b0.ns_jft_in1k"
    EPOCHS = 2
    N_FOLDS = 2
    subset_size = 5


class PytorchLightningConfig(Config):
    # Override or extend settings specific to PyTorch Lightning
    OUTPUT_DIR = 'pl-results'
    BATCH_SIZE = 64


class ColabConfig(Config):
    pass


def get_config(env='local'):
    if env == 'local':
        return LocalConfig()
    elif env == 'pl':
        return PytorchLightningConfig()
    elif env == 'colab':
        return ColabConfig()
    else:
        raise ValueError(f"Unknown environment: {env}")


# Usage
config = get_config(env='local')
print(f"Running with config: {config.MODEL_NAME}, device: {config.device}")


def set_seed(seed=config.SEED):
    """set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'set seed {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transforms_train = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=config.AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=config.AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=config.AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=15, border_mode=0, p=config.AUG_PROB),
    A.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1]),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64,
                    min_holes=1, min_height=8, min_width=8, p=config.AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])
