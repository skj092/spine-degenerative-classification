import numpy as np
from pathlib import Path
import pandas as pd
from conf import transforms_train
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from conf import config


# Dataset and DataLoader Class
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
                # print(f'failed to load on {st_id}, Sagittal T1')
                pass

        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'./cvt_png/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
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
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass

        assert np.sum(x) > 0

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)

        return x, label


def create_dataloader(df, phase, transform, batch_size, shuffle, drop_last, num_workers):
    dataset = RSNA24Dataset(df, phase=phase, transform=transform)
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
    rd = Path(__file__).parent.parent
    df = pd.read_csv(rd / 'data/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    tmp_ds = RSNA24Dataset(df, phase='train', transform=transforms_train)
    tmp_dl = DataLoader(
        tmp_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    for i, (x, t) in enumerate(tmp_dl):
        if i == 5:
            break
        print('x stat:', x.shape, x.min(), x.max(), x.mean(), x.std())
        print(t, t.shape)
        y = x.numpy().transpose(0, 2, 3, 1)[0, ..., :3]
        y = (y + 1) / 2
        plt.imshow(y)
        plt.savefig(f'./tmp_{i}.png')
        plt.show()
        print('y stat:', y.shape, y.min(), y.max(), y.mean(), y.std())
        print()
    plt.close()
    del tmp_ds, tmp_dl
