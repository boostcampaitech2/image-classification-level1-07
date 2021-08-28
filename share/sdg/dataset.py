from data_preprocessing import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize, ToPILImage, RandomHorizontalFlip
from torchvision.transforms.functional import crop
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import os
from PIL import Image



class TrainDataset(Dataset):
    def __init__(self, img_paths, label, transform, train = True):
        self.img_paths = img_paths
        self.transform = transform
        self.label = label
        self.classes = pd.Series(self.label).unique()
        self.train = train
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        y = self.label[index] 
        if self.transform:
            if self.train:
                if 'out_data' in self.img_paths[index]:
                    image = self.transform['train_crop'](image=np.array(image))['image']
                else:
                    image = self.transform['train'](image=np.array(image))['image']
            else:
                image = self.transform['val'](image=np.array(image))['image']
        return image, torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform['val'](image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)


def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            # Resize(img_size[0], img_size[1], p=1.0),
            Resize(384,384, p=1.0),
            CenterCrop(288,288),
            # Resize(288,288, p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
        transformations['train_crop'] = Compose([
            # Resize(img_size[0], img_size[1], p=1.0),
            Resize(288,288, p=1.0),
            # CenterCrop(288,288),
            # Resize(288,288, p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    if 'val' in need:
        transformations['val'] = Compose([
            # Resize(img_size[0], img_size[1]),
            # Resize(288,288),
            Resize(384,384, p=1.0),
            CenterCrop(288,288),
            Normalize(mean=mean, std=std, max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations


def generate_dataset_parallel(train_info, valid_info):
    train_image, train_label_age, class_num_a = generate_label(train_info ,option='age')
    _, train_label_gender, class_num_g = generate_label(train_info ,option='gender')
    _, train_label_mask, class_num_m = generate_label(train_info ,option='mask')
    _, train_label_total, class_num_t = generate_label(train_info ,option='normal')
    train_label = []
    for i in range(len(train_label_mask)):
        train_label.append((train_label_mask[i], train_label_gender[i], train_label_age[i], train_label_total[i]))

    valid_image, valid_label_age, _ = generate_label(valid_info ,option='age')
    _, valid_label_gender, _ = generate_label(valid_info ,option='gender')
    _, valid_label_mask, _ = generate_label(valid_info ,option='mask')
    _, valid_label_total, _ = generate_label(valid_info ,option='normal')
    valid_label = []
    for i in range(len(valid_label_mask)):
        valid_label.append((valid_label_mask[i], valid_label_gender[i], valid_label_age[i], valid_label_total[i]))
    transform = get_transforms(need = ('train', 'val'))

    Dataset_Train = TrainDataset(img_paths = train_image, label = train_label, transform = transform, train = True)
    Dataset_Valid = TrainDataset(img_paths = valid_image, label = valid_label, transform = transform, train = False)
    return Dataset_Train, Dataset_Valid, (class_num_m, class_num_g, class_num_a, class_num_t)

def generate_dataloader_parallel(train_info, valid_info, batch_size):
    Dataset_Train, Dataset_Valid, class_num = generate_dataset_parallel(train_info, valid_info)
    dataloaders_train = DataLoader(dataset=Dataset_Train, batch_size=batch_size, shuffle=True, num_workers = 2)
    dataloaders_valid = DataLoader(dataset=Dataset_Valid, batch_size=batch_size, shuffle=False, num_workers = 2)
    return dataloaders_train, dataloaders_valid, class_num


def generate_testdataset(image_path):
    transform = get_transforms(need = ('val'))
    Dataset_Test = TestDataset(img_paths = image_path, transform = transform)
    dataloaders_test = DataLoader(Dataset_Test, shuffle=False)
    return dataloaders_test


