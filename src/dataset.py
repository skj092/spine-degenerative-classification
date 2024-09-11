import numpy as np
import pandas as pd
import json
from conf import get_transforms
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# Dataset and DataLoader Class
class RSNA24Dataset(Dataset):
    def __init__(self, df, config, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
        self.data_dir = config.DATA_DIR
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((512, 512, self.config.IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)

        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                # p = f'./cvt_png/{st_id}/Sagittal T1/{i:03d}.png'
                p = f"{self.data_dir}/cvt_png/{st_id}/Sagittal T1/{i:03d}.png"
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T1')
                pass

        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                # p = f'./cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                p = f"{self.data_dir}/cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png"
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass

        # Axial T2
        # axt2 = glob(f'./cvt_png/{st_id}/Axial T2/*.png')
        axt2 = glob(f"{self.data_dir}/cvt_png/{st_id}/Axial T2/*.png")
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
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass

        assert np.sum(x) > 0

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)

        return x, label


def create_dataloader(df, phase, transform, batch_size, shuffle, drop_last, num_workers, config=None):
    dataset = RSNA24Dataset(
        df, phase=phase, transform=transform, config=config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch_factor=2
    )


if __name__ == "__main__":
    config = json.load(open('configs/local_config.json'))
    config = Config(config)
    df = pd.read_csv(f'{config.CSV_PATH}/train.csv')
    subset_size = config.subset_size
    if subset_size:
        print(f"Using subset of size: {subset_size}")
        df = df.sample(n=subset_size, random_state=config.SEED)
    print(f"DataFrame shape: {df.shape}")
    df.fillna(-100, inplace=True)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.map(lambda x: label2id.get(x, x))
    transforms_train, transforms_val = get_transforms(config)

    train_loader = create_dataloader(
        df, 'train', transforms_train, config.BATCH_SIZE, True, False, 4, config=config)
    val_loader = create_dataloader(
        df, 'val', transforms_val, config.BATCH_SIZE, False, False, 4, config=config)
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
