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
                image = self.transform['train'](image=np.array(image))['image']
            else:
                image = self.transform['val'](image=np.array(image))['image']
        return image, torch.tensor(y, dtype=torch.long)

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
            Resize(img_size[0], img_size[1], p=1.0),
            Resize(288,288,p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            Resize(288,288,p=1.0),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations

def generate_dataset(train_info, option):
    train_image, train_label, class_num = generate_label(train_info ,option)
    transform = get_transforms(need = ('train', 'val'))
    Dataset_Train = TrainDataset(img_paths = train_image, label = train_label, transform = transform, train = True)
    Dataset_Valid = TrainDataset(img_paths = train_image, label = train_label, transform = transform, train = False)
    return Dataset_Train, Dataset_Valid, class_num

def generate_dataloader(train_info, option, batch_size):
    Dataset_Train, Dataset_Valid, class_num = generate_dataset(train_info, option)
    dataloaders_train = DataLoader(dataset=Dataset_Train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(0, len(Dataset_Train) * 4//5)), num_workers = 2)
    dataloaders_valid = DataLoader(dataset=Dataset_Valid, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(len(Dataset_Valid) * 4//5, len(Dataset_Valid))), num_workers = 2)
    return dataloaders_train, dataloaders_valid, class_num


def generate_testdataset(image_path):
    transform = get_transforms(need = ('val'))
    Dataset_Test = TestDataset(img_paths = image_path, transform = transform)
    dataloaders_test = DataLoader(Dataset_Test, shuffle=False)
    return dataloaders_test




