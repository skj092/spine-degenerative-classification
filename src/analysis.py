import os
import glob
import random
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import albumentations as A
import cv2

import torch
import pydicom


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype("uint8")


def load_dicom_stack(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray(
        [float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]
                     ).astype("float")[idx]
    array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    return {"array": convert_to_8bit(array), "positions": ipp, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float")}


resize_transform = A.Compose([
    A.LongestMaxSize(
        max_size=256, interpolation=cv2.INTER_CUBIC, always_apply=True),
    A.PadIfNeeded(min_height=256, min_width=256,
                  border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
])


def angle_of_line(x1, y1, x2, y2):
    return math.degrees(math.atan2(-(y2-y1), x2-x1))


def plot_img(img, coords_temp):
    # Plot img
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    h, w = img.shape

    # Kepoints as pairs
    p = coords_temp.groupby("level") \
        .apply(lambda g: list(zip(g['relative_x'], g['relative_y'])), include_groups=False) \
        .reset_index(drop=False, name="vals")

    # Plot keypoints
    for _, row in p.iterrows():
        level = row['level']
        x = [_[0]*w for _ in row["vals"]]
        y = [_[1]*h for _ in row["vals"]]
        ax.plot(x, y, marker='o')
    ax.axis('off')
    plt.savefig("tmp/img.png")
    # plt.show()


def plot_5_crops(img, coords_temp):
    # Create a figure and axis for the grid
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1]*5)

    # Plot the crops
    p = coords_temp.groupby("level").apply(lambda g: list(zip(
        g['relative_x'], g['relative_y'])), include_groups=False).reset_index(drop=False, name="vals")
    for idx, (_, row) in enumerate(p.iterrows()):
        # Copy of img
        img_copy = img.copy()
        h, w = img.shape

        # Extract Keypoints
        level = row['level']
        vals = sorted(row["vals"], key=lambda x: x[0])
        a, b = vals
        a = (a[0]*w, a[1]*h)
        b = (b[0]*w, b[1]*h)

        # Rotate
        rotate_angle = angle_of_line(a[0], a[1], b[0], b[1])
        transform = A.Compose([
            A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

        t = transform(image=img_copy, keypoints=[a, b])
        img_copy = t["image"]
        a, b = t["keypoints"]

        # Crop + Resize
        img_copy = crop_between_keypoints(img_copy, a, b)
        img_copy = resize_transform(image=img_copy)["image"]

        # Plot
        ax = plt.subplot(gs[idx])
        ax.imshow(img_copy, cmap='gray')
        ax.set_title(level)
        ax.axis('off')
    plt.savefig("tmp/5_crops.png")
    # plt.show()


def crop_between_keypoints(img, keypoint1, keypoint2):
    h, w = img.shape
    x1, y1 = int(keypoint1[0]), int(keypoint1[1])
    x2, y2 = int(keypoint2[0]), int(keypoint2[1])

    # Calculate bounding box around the keypoints
    left = int(min(x1, x2))
    right = int(max(x1, x2))
    top = int(min(y1, y2) - (h * 0.1))
    bottom = int(max(y1, y2) + (h * 0.1))

    # Crop the image
    return img[top:bottom, left:right]


if __name__ == "__main__":
    SEED = 10
    N = 2
    image_dir = "data/train_images/"
    data_dir = "data"
    tmp_dir = "tmp"
    # Load series_ids
    dfd = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
    dfd = dfd[dfd.series_description == "Sagittal T2/STIR"]
    dfd = dfd.sample(frac=1, random_state=SEED).head(N)

    # Load coords
    coords = pd.read_csv(f"{tmp_dir}/coords_rsna_improved.csv")
    coords = coords.sort_values(
        ["series_id", "level", "side"]).reset_index(drop=True)
    coords = coords[["series_id", "level", "side", "relative_x", "relative_y"]]

    # Plot samples
    for idx, row in dfd.iterrows():
        try:
            print("-"*25, " STUDY_ID: {}, SERIES_ID: {} ".format(row.study_id,
                  row.series_id), "-"*25)
            sag_t2 = load_dicom_stack(os.path.join(image_dir, str(
                row.study_id), str(row.series_id)), plane="sagittal")

            # Img + Coords
            img = sag_t2["array"][len(sag_t2["array"])//2]
            coords_temp = coords[coords["series_id"] == row.series_id].copy()

            # Plot
            plot_img(img, coords_temp)
            plot_5_crops(img, coords_temp)

        except Exception as e:
            print(e)
            pass
