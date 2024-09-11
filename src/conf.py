import albumentations as A
import torch
import os
import random
import numpy as np


def set_seed(seed=42):
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


def get_transforms(config):
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
    return transforms_train, transforms_val
