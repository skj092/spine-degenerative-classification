import argparse
import json
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
import wandb


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.records = self.load_coords(df)

    def load_coords(self, df):
        # Convert to dict
        d = df.groupby("series_id")[["relative_x", "relative_y"]].apply(
            lambda x: list(x.itertuples(index=False, name=None)))
        records = {}
        for i, (k, v) in enumerate(d.items()):
            records[i] = {"series_id": k, "label": np.array(v).flatten()}
            assert len(v) == 5

        return records

    def pad_image(self, img):
        n = img.shape[-1]
        if n >= self.cfg.n_frames:
            start_idx = (n - self.cfg.n_frames) // 2
            return img[:, :, start_idx:start_idx + self.cfg.n_frames]
        else:
            pad_left = (self.cfg.n_frames - n) // 2
            pad_right = self.cfg.n_frames - n - pad_left
            return np.pad(img, ((0, 0), (0, 0), (pad_left, pad_right)), 'constant', constant_values=0)

    def load_img(self, source, series_id):
        fname = os.path.join(
            self.cfg.data_dir, "processed_{}/{}.npy".format(source, series_id))
        img = np.load(fname).astype(np.float32)
        img = self.pad_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = (img / 255.0)
        return img

    def __getitem__(self, idx):
        d = self.records[idx]
        label = d["label"]
        source = d["series_id"].split("_")[0]
        series_id = "_".join(d["series_id"].split("_")[1:])

        img = self.load_img(source, series_id)
        return {
            'img': img,
            'label': label,
        }

    def __len__(self,):
        return len(self.records)


# Utils
def batch_to_device(batch, device, skip_keys=[]):
    batch_dict = {}
    for key in batch:
        if key in skip_keys:
            batch_dict[key] = batch[key]
        else:
            batch_dict[key] = batch[key].to(device)
    return batch_dict


def visualize_prediction(batch, pred, epoch):

    mid = cfg.n_frames//2

    # Plot
    for idx in range(1):

        # Select Data
        img = batch["img"][idx, mid, :, :].cpu().numpy()*255
        cs_true = batch["label"][idx, ...].cpu().numpy()*256
        cs = pred[idx, ...].cpu().numpy()*256

        coords_list = [("TRUE", "lightblue", cs_true), ("PRED", "orange", cs)]
        text_labels = [str(x) for x in range(1, 6)]

        # Plot coords
        fig, axes = plt.subplots(1, len(coords_list), figsize=(10, 4))
        fig.suptitle("EPOCH: {}".format(epoch))
        for ax, (title, color, coords) in zip(axes, coords_list):
            ax.imshow(img, cmap='gray')
            ax.scatter(coords[0::2], coords[1::2], c=color, s=50)
            ax.axis('off')
            ax.set_title(title)

            # Add text labels near the coordinates
            for i, (x, y) in enumerate(zip(coords[0::2], coords[1::2])):
                if i < len(text_labels):  # Ensure there are enough labels
                    ax.text(x + 10, y, text_labels[i], color='white',
                            fontsize=15, bbox=dict(facecolor='black', alpha=0.5))

        fig.suptitle("EPOCH: {}".format(epoch))
        plt.show()
    #   plt.close(fig)
    return


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


if __name__ == "__main__":
    # Load the config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        default="configs/pretrain_config.json")
    config_dict = json.load(open(parser.parse_args().cfg))
    cfg = Config(config_dict)
    if cfg.USE_WANDB:
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_API_KEY"] = cfg.WANDB_API_KEY
        wandb.init(project="rsna24", config=cfg,
                   name="RSNA24 Backbone Pretraining")

    # set device
    cfg.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed=cfg.seed)  # Makes results reproducable
    # Load metadata
    df = pd.read_csv(f"{cfg.data_dir}/coords_pretrain.csv")
    df = df.sort_values(["source", "filename", "level"]).reset_index(drop=True)
    df["filename"] = df["filename"].str.replace(".jpg", ".npy")
    df["series_id"] = df["source"] + "_" + df["filename"].str.split(".").str[0]

    print("----- IMGS per source -----")
    ds = PreTrainDataset(df, cfg)
    print(f"Total Samples: {len(ds)}")

    # Plot a Single Sample
    print("---- Sample Shapes -----")
    for k, v in ds[0].items():
        print(k, v.shape)

    # Model Training
    # Dataframes
    train_df = df[df["source"] != "spider"]
    val_df = df[df["source"] == "spider"]
    print("TRAIN_SIZE: {}, VAL_SIZE: {}".format(
        len(train_df)//5, len(val_df)//5))

    # Datasets + Dataloaders
    train_ds = PreTrainDataset(train_df, cfg)
    val_ds = PreTrainDataset(val_df, cfg)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model
    # tf_efficientnet_b3.ns_jft_in1k
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    # Use all the GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(cfg.device)

    # Loss / Optim
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs+1):

        # Train Loop
        loss = torch.tensor([0.]).float().to(cfg.device)
        if epoch != 0:
            model = model.train()
            for batch in tqdm(train_dl):
                batch = batch_to_device(batch, cfg.device)
                optimizer.zero_grad()

                x_out = model(batch["img"].float())
                x_out = torch.sigmoid(x_out)

                loss = criterion(x_out, batch["label"].float())
                if cfg.USE_WANDB:
                    wandb.log({"train_loss": loss.item()})
                loss.backward()
                optimizer.step()

        # Validation Loop
        val_loss = 0
        with torch.no_grad():
            model = model.eval()
            for batch in tqdm(val_dl):
                batch = batch_to_device(batch, cfg.device)

                pred = model(batch["img"].float())
                pred = torch.sigmoid(pred)

                val_loss += criterion(pred, batch["label"].float()).item()
                if cfg.USE_WANDB:
                    wandb.log({"val_loss": val_loss})
            val_loss /= len(val_dl)

        print(
            f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
    print("Training complete...")

    f = "{}_{}.pt".format(cfg.backbone, cfg.seed)
    torch.save(model.state_dict(), f)
    print("Saved weights: {}".format(f))

    # Load backbone for RSNA 2024 task
    model = timm.create_model('resnet18', pretrained=True, num_classes=75)
    model = model.to(cfg.device)
    load_weights_skip_mismatch(model, f, cfg.device)
