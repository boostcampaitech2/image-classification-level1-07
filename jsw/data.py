import os
from re import L
import pandas as pd
from PIL import Image

import torch
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import ImageFolder

from albumentations import *
from albumentations.pytorch.transforms import *

mean_ = [0.485, 0.456, 0.406]
std_ = [0.229, 0.224, 0.225]

def get_dataset(root, csv_dir, train=True, cutmix=False):
    if train:
        if cutmix:
            #? weak augmentation pipeline when cutmix applied
            transform = Compose([
                HorizontalFlip(),
                Resize(width=224, height=224),
                Normalize(mean=mean_, std=std_),
                ToTensorV2()
            ])
            #? strong augmentation pipeline when cutmix NOT applied
            aug_transform = Compose([
                Resize(width=224,height=224),
                HorizontalFlip(),
                RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                Rotate(limit=5),   
                GaussNoise(p=0.2),
                Normalize(mean=mean_, std=std_),
                ToTensorV2(),
            ])
            return CustomDataset(root, csv_dir, train, transform, aug_transform)
        else:
            #? augmentation pipeline when no cutmix
            aug_transform = Compose([
                Resize(width=224,height=224),
                HorizontalFlip(),
                Rotate(limit=5),
                Normalize(mean=mean_, std=std_),
                ToTensorV2(),
            ])
            return CustomDataset(root, csv_dir, train, aug_transform)
    else:
        #? Validation transformation pipeline
        transform = Compose([
                Resize(width=224,height=224),
                Normalize(mean=mean_, std=std_),
                ToTensorV2(),
        ])
        return CustomDataset(root, csv_dir, train, transform)
    
def get_unlabeled_dataset(root, train=True):
    transform = Compose([
                Resize(width=224,height=224),
                Normalize(mean=mean_, std=std_),
                ToTensorV2(),
        ])
    dataset = UnlabeledDataset(root, transform)
    return dataset

def get_combined_dataset(root, unlabeled_root, csv, cutmix=False):
    transform = Compose([
        HorizontalFlip(),
        Resize(width=224,height=224),
        Normalize(mean=mean_, std=std_, always_apply=True),
        ToTensorV2(),
    ])
    aug = Compose([
        Resize(height=224, width=224),
        HorizontalFlip(),
        Rotate(limit=5),
        CLAHE(p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        GaussNoise(p=0.5),
        Normalize(mean=mean_,std=std_,always_apply=True),
        ToTensorV2(),
    ],p=1.0)
    if not cutmix:
        return CombinedDataset(root, unlabeled_root, csv, transform=aug)
    else:
        return CombinedDataset(root, unlabeled_root, csv, transform=transform, aug=aug)
    
def get_data_loader(dataset,
                    sampler=None,
                    batch_size=64,
                    num_workers=2,
                    train=True):
    
    #? if no sampler given, random sampler
    if sampler is None:
        sampler = RandomSampler(dataset)
    
    if train:
        loader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            num_workers=num_workers)
    else:
        loader = DataLoader(dataset,
                            shuffle=False,
                            num_workers=num_workers)
        
    return loader
        
class CustomDataset(Dataset):
    def __init__(self, root, csv_dir, train=True, ori_transform=None, aug_transform=None, cutmix=False):
        self.root = root
        self.train = train
        self.cutmix = cutmix
        self.ori_transform = ori_transform
        self.aug_transform = aug_transform
        
        self.csv_file = pd.read_csv(os.path.join(self.root, csv_dir))
    
    def __len__(self):
        return len(self.csv_file['path'])
    
    def __getitem__(self, index):
        path, label = self.csv_file.iloc[index]
        X = Image.open(path)
        y = label
        if self.ori_transform is not None:
                ori = self.ori_transform(image=np.array(X))['image']
        if self.aug_transform is not None:
                aug = self.aug_transform(image=np.array(X))['image']    
                    
        if self.train:
            if self.cutmix:
                return aug, ori, y
            else:
                return ori, y
        else:
            return ori, int(y)

class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.data = ImageFolder(self.root)
        self.label = [0 for _ in range(len(self.data))]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X, _ = self.data[index]
        
        if self.transform is not None:
            return self.transform(image=np.array(X))['image'], self.label[index]
        else:
            return X, self.label[index]
        
class CombinedDataset(Dataset):
    def __init__ (self, root, unlabeled_root, csv_dir, train=True, transform=None, aug=None):
        self.root = root
        self.csv_file = pd.read_csv(os.path.join(root, csv_dir))
        self.unlabeled_data = ImageFolder(unlabeled_root)
        
        self.labeled_len = len(self.csv_file)
        self.unlabeled_len = len(self.unlabeled_data)
        
        self.unlabeled_label = [0 for _ in range(self.unlabeled_len)]
        self.unlabeled_label = torch.LongTensor(self.unlabeled_label)
        
        self.aug = aug
        self.transform = transform
        
    def __len__ (self):
        return self.unlabeled_len + self.labeled_len
    
    def __getitem__(self, index):
        if index < self.labeled_len:
            path, label = self.csv_file.iloc[index]
            X = Image.open(path)
            y = torch.scalar_tensor(label, dtype=torch.long)
        else:
            X, _ = self.unlabeled_data[index-self.labeled_len]
            y = self.unlabeled_label[index-self.labeled_len]
            
        if self.transform is not None and self.aug is None:
            X = self.transform(image=np.array(X))['image']
            return X, y
        elif self.transform is not None and self.aug is None:
            ori = self.transform(image=np.array(X))['image']
            aug = self.aug(image=np.array(X))['image']
            return aug, ori, y
        return X, y
    
    def set_labels(self, labels):
        for i, label in enumerate(labels):
            self.unlabeled_label[i] = label