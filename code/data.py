import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms 
from torchvision.datasets import ImageFolder

from PIL import Image
import pandas as pd
import numpy as np
import os

def split_dataset(data_path='/opt/ml/input/data/train', csv_name = 'train_total_facecrop.csv', random_seed = 42):
    # train_dir = '/opt/ml/input/data/train'
    traindf = pd.read_csv(os.path.join(data_path, csv_name))

    X = traindf['path']
    y = traindf['class']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=random_seed, stratify=y)

    return X_train.values, X_valid.values, y_train.values, y_valid.values

class TrainDataset(Dataset):
    def __init__(self, path, transform, train):
        self.path = path
        self.transform = transform
        self.csv = pd.read_csv(path)
        self.train = train
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()
            
        path = self.csv['img_path'].values
        labels = self.csv['person_label'].values
        X = Image.open(path[index])
        y = labels[index]

        if self.transform:
            if self.train == True:
                aug_img = self.transform['train'](X)
                return aug_img, torch.tensor(y, dtype=torch.long)   
            
            else:
                X = self.transform['val'](X)
                return X, torch.tensor(y, dtype=torch.long)  

class TrainDataset_cutmix(Dataset):
    def __init__(self, path, transform, train):
        self.path = path
        self.transform = transform
        self.csv = pd.read_csv(path)
        self.train = train
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()
            
        path = self.csv['img_path'].values
        labels = self.csv['person_label'].values
        X = Image.open(path[index])
        y = labels[index]

        if self.transform:
            if self.train == True:
                aug_img = self.transform['train'](X)
                ori_img = self.transform['val'](X)
                return aug_img, ori_img, torch.tensor(y, dtype=torch.long)   
            
            else:
                X = self.transform['val'](X)
                return X, torch.tensor(y, dtype=torch.long)   

class GivenTrainDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        # self.df = pd.read_csv(os.path.join(self.path, 'train_revised.csv'))
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()
        X = Image.open(self.paths[index])
        y = self.labels[index]

        if self.transform:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)   


class TestDataset(Dataset):
    def __init__(self, eval_paths, transform):
        self.eval_paths = eval_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.eval_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.eval_paths[index])
        
        if self.transform:
            img = self.transform(img)
            
        return img
    
# unlabeled dataset class
class UnlabeledDataset(Dataset):
    def __init__(self, root, transform, train=True):
        self.root = root
        self.train = train
        
        self.data = ImageFolder(self.root)
        
        self.label = [0 for i in range(len(self.data))]
        
        self.transform = transform
        # self.augmentations = transforms.Compose([
        #     transforms.RandomRotation(5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.CenterCrop(384),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_, std=std_)
        # ])
        # self.normalize = transforms.Compose([
        #     transforms.CenterCrop(384),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_, std=std_)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X, _ = self.data[index]
        
        # if self.train:
        #     return self.augmentations(X), self.label[index]
        # else:
        #     return self.normalize(X), self.label[index]
        if self.transform:
            if self.train:
                return self.transform['train'](X), self.label[index]
            else:
                return self.transform['val'](X), self.label[index]
            
class CombineDataset(Dataset):
    def __init__(self, ori_path,  u_path, transform, train = True):
        self.train = train
        self.ori_path = ori_path
        self.u_path = u_path
        self.transform = transform
        self.l_data = pd.read_csv(ori_path)
        self.u_data = ImageFolder(u_path)
        self.u_len = len(self.u_data)
        self.l_len = len(self.l_data)
        
        u_label = [0 for _ in range(self.u_len)]
        self.u_label = torch.LongTensor(u_label)
        
    def __len__(self):
        return self.l_len + len(self.u_label)
        
    def __getitem__(self, index):
        path = self.l_data['img_path'].values
        labels = self.l_data['person_label'].values
        
        if index < self.l_len:
            X = Image.open(path[index])
            y = labels[index] 
        else:
            idx = index - self.l_len
            X = self.u_data[idx]
            y = self.u_label[idx]
            
        if self.train==True:
            X = self.transform['train'](X)
        else:
            X = self.transform['val'](X)
        return X, torch.tensor(y, dtype=torch.long)
    
    def set_label(self, label):
        self.u_label = torch.LongTensor(label)

def get_transforms(need=('train', 'val'), img_size=(224,224), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    """
    train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
    Args:
        need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
        img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

    Returns:
        transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
    """
    transformations = {}
    if 'train' in need:
        transformations['train'] = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Normalize(mean=mean, std=std),
        ])
    if 'val' in need:
        transformations['val'] = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transformations

