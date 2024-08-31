import pandas as pd
from pathlib import Path
from PIL import Image
import pydicom
import os
from tqdm import tqdm
import glob
import re
import numpy as np
import cv2


# rd = Path("/home/sonujha/rnd/spine-degenerative-classification")
rd = Path('/teamspace/studios/this_studio/spine-degenerative-classification')


# Function to convert the DICOM files to PNG
def convert_to_png(src_path, dest_path):
    # Convert the DICOM files to PNG
    image = pydicom.dcmread(src_path)
    image = image.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    assert image.shape == (512, 512)
    cv2.imwrite(dest_path, image)


# Text to Integer
def atoi(text):
    '''Converts text to integer
    example 1: atoi('123') -> 123
    example 2: atoi('abc') -> 'abc'
    '''
    return int(text) if text.isdigit() else text


# Natural Sorting
def natural_keys(text):
    '''Sorts the text in human order
    example: ['a1', 'a10', 'a2'] -> ['a1', 'a2', 'a10']
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Series description

# This file contains the "study_id", "series_id" and "series_description" of the images
df = pd.read_csv(
    rd / "data/train_series_descriptions.csv")

unique_series_description = df["series_description"].unique()
# ['Sagittal T2/STIR', 'Sagittal T1', 'Axial T2']

unique_study_id = df["study_id"].unique()
# 1975 unique study_id


for idx, si in enumerate(tqdm(unique_study_id, total=len(unique_study_id))):
    pdf = df[df['study_id'] == si]
    for ds in unique_series_description:
        ds_ = ds.replace('/', '_')
        pdf_ = pdf[pdf['series_description'] == ds]
        os.makedirs(f'cvt_png/{si}/{ds_}', exist_ok=True)
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(
                f'{rd}/data/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)

        if len(allimgs) == 0:
            print(si, ds, 'has no images')
            continue

        if ds == 'Axial T2':
            for j, impath in enumerate(allimgs):
                dst = f'cvt_png/{si}/{ds}/{j:03d}.png'
                convert_to_png(impath, dst)

        elif ds == 'Sagittal T2/STIR':

            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'cvt_png/{si}/{ds_}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                convert_to_png(allimgs[ind2], dst)

            assert len(glob.glob(f'cvt_png/{si}/{ds_}/*.png')) == 10

        elif ds == 'Sagittal T1':
            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'cvt_png/{si}/{ds}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                convert_to_png(allimgs[ind2], dst)

            assert len(glob.glob(f'cvt_png/{si}/{ds}/*.png')) == 10
