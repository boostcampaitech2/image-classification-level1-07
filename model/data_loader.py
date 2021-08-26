import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

import cv2
import albumentations as A
from albumentations.augmentations.transforms import CLAHE
from albumentations.pytorch import ToTensorV2


def get_data_transform(width: int, height: int, max_size: int):
    assert width, 'width required!'
    assert height, 'height required!'
    assert max_size, 'max_size required!'

    data_transform = {
        'train': A.Compose([
            # A.CLAHE(), Random 하게
            # A.Blur(),
            # A.HueSaturationValue(),
            # A.JpegCompression(),
            # A.SmallestMaxSize(max_size=416),  # efficientnetv2_rw_m
            # A.RandomCrop(width=320, height=320),  # efficientnetv2_rw_m
            A.SmallestMaxSize(max_size=max_size),  # efficientnetv2_rw_s
            A.RandomCrop(width=width, height=height),  # efficientnetv2_rw_s
            # A.RandomCrop(width=224, height=224),  # ResNet18
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.GaussNoise(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),

        'val': A.Compose([
            # A.SmallestMaxSize(max_size=416),  # efficientnetv2_rw_m
            A.SmallestMaxSize(max_size=max_size),  # efficientnetv2_rw_s
            # A.SmallestMaxSize(max_size=320),
            # A.CenterCrop(width=224, height=224),  # ResNet18
            # A.CenterCrop(width=320, height=320),  # efficientnetv2_rw_m
            A.CenterCrop(width=width, height=height),  # efficientnetv2_rw_s
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    }
    return data_transform


## https://albumentations.ai/docs/examples/pytorch_classification/
# train_transform = A.Compose([
#     A.SmallestMaxSize(max_size=160),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#     A.RandomCrop(height=128, width=128),
#     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])

class FaceMaskDataset(Dataset):
    """Ai Tech 2th, P Stage, Week 4~5 Face mask detection task, dataset."""

    def __init__(self, csv_file, transform=None, target=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            # root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target = target
        _df = pd.read_csv(csv_file)
        self.mask = _df['stem'].map(
            {'normal': 2,
            'incorrect_mask': 1,
            'mask1': 0,
            'mask2': 0,
            'mask3': 0,
            'mask4': 0,
            'mask5': 0}).astype(int).to_list()
        _df.loc[_df['age'] < 30, '_age'] = 0
        _df.loc[(_df['age'] >= 30) & (_df['age'] < 60), '_age'] = 1
        _df.loc[_df['age'] >= 60, '_age'] = 2

        _df.loc[_df['age'] < 30, 'person'] = 0
        _df.loc[(_df['age'] >= 30) & (_df['age'] < 60), 'person'] = 1
        _df.loc[_df['age'] >= 60, 'person'] = 2
        _df.loc[_df['gender'] == 'female', 'person'] += 3

        _df['_age'] = _df['_age'].astype(int)
        self.age = _df['_age'].to_list()
        self.gender = _df['gender'].map({'male': 0, 'female': 1}).to_list()
        self.person = _df['person'].to_list()
        self.img_path_list = _df['img_path'].to_list()
        self.label_list = _df['label'].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.target:
            if self.target == 'gender':
                _label = self.gender[idx]
            elif self.target == 'age':
                _label = self.age[idx]
            elif self.target == 'mask':
                _label = self.mask[idx]
            elif self.target == 'person':
                _person = self.person[idx]
                _mask = self.mask[idx]
                _label = self.label_list[idx]
                _label = torch.tensor([_person, _mask, _label], dtype=torch.long)
        else:
            _age = self.age[idx]
            _gender = self.gender[idx]
            _mask = self.mask[idx]
            _label = self.label_list[idx]
            _label = torch.tensor([_age, _gender, _mask, _label], dtype=torch.long)

        _img_path = self.img_path_list[idx]
        _image = cv2.imread(_img_path)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            _image = self.transform(image=_image)['image']
        return _image, _label


if __name__ == '__main__':
    pass