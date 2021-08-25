from numpy.core.numeric import zeros_like
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms.transforms import CenterCrop, ToTensor

class_nums = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109]
class_weights = [1./c for c in class_nums]
class_weights = torch.FloatTensor(class_weights)

mean_ = [0.485, 0.456, 0.406]
std_ = [0.229, 0.224, 0.225]

def get_loader(dataset, batch_size = 64, num_workers=4):
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return loader

def get_train_val_loader(l_train_set, l_val_set, batch_size=64, num_workers=2):
    weights = [class_weights[y] for _, y in l_train_set]
    weights = torch.FloatTensor(weights)
    
    l_train_loader = DataLoader(
        l_train_set,
        sampler=WeightedRandomSampler(weights, len(weights)),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    l_val_loader = DataLoader(
        l_val_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return l_train_loader, l_val_loader

def get_labeled_datasets(root, csv_file, train):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_, std=std_)])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_, std=std_)])
    if train:
        dataset = GivenDataset(root, csv_file, train=train, transform=transform_labeled)
    else:
        dataset = GivenDataset(root, csv_file, train=train, transform=transform_val)
    return dataset

def get_unlabeled_datasetes(root, train):
    dataset = UnlabeledDataset(root, train)
    return dataset

class UnlabeledDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        
        self.data = ImageFolder(self.root)
        
        self.label = [0 for i in range(len(self.data))]
        
        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=324,
                                    padding=int(324*0.125),
                                    padding_mode='reflect')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=324,
                                    padding=int(324*0.125),
                                    padding_mode='reflect')])
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)
        ])
        self.normalize = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X, _ = self.data[index]
        
        if self.train:
            return self.augmentations(X), self.label[index]
        else:
            return self.normalize(X), self.label[index]

class GivenDataset(Dataset):
    def __init__(self, root, csv_dir, train=True, transform=None, target_transform=None):
        self.root = root
        self.csv_file = pd.read_csv(os.path.join(root, csv_dir))
        self.train = train
        self.transform = transform
        


    def __len__(self):
        return len(self.csv_file['id'])

    def __getitem__(self, idx):
        _, _, _, _, path, label = self.csv_file.iloc[idx]
        X = Image.open(path)
        y = label
        if self.transform is not None:
            X = self.transform(X)
        return X, y


class CombinedDataset(Dataset):
    def __init__(self, labeled_root, csv_dir, unlabeled_root, train=True):
        self.train = train
        
        self.unlabeled_data = ImageFolder(unlabeled_root)
        self.labeled_data = pd.read_csv(os.path.join(labeled_root, csv_dir))
        
        self.u_len = len(self.unlabeled_data)
        self.l_len = len(self.labeled_data['id'])
        
        u_label = [0 for _ in range(self.u_len)]
        self.u_label = torch.LongTensor(u_label)
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)
        ])
        self.normalize = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_)])
        
    def __len__(self):
        return self.u_len + self.l_len
    
    def __getitem__(self, idx):
        X = None
        y = None
        if idx < self.l_len:
            _, _, _, _, path, label = self.labeled_data.iloc[idx]
            X = Image.open(path)
            y = torch.scalar_tensor(label, dtype=torch.long)
        else:
            X, _ = self.unlabeled_data[idx-self.l_len]
            y = self.u_label[idx-self.l_len]
            
        if self.train:
            X = self.augmentations(X)
        else:
            X = self.normalize(X)
        return X, y
        
    def set_unlabeled_label(self, index, label):
        self.u_label[index] = label