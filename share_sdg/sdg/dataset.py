import torch
from torch.utils.data import Dataset
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image



class TrainDataset(Dataset):
    def __init__(self, df, train = True):
        self.df = df
        transformations = { 'train' : Compose([
            Resize(224,224, p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=(0.55720607, 0.47626135, 0.44402045), std=(0.2195448 , 0.21720374 ,0.22056428), max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0),
            'val' : Compose([
            Resize(224,224, p=1.0),
            Normalize(mean=(0.55720607, 0.47626135, 0.44402045), std=(0.2195448 , 0.21720374 ,0.22056428), max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0) 
        }
        self.transform = transformations
        self.label = self.df['class']
        self.train = train
    
    def __getitem__(self, index):
        image = Image.open(self.df['img_path'][index])
        y = self.label[index] 
        if self.transform:
            if self.train:
                aug_image = self.transform['train'](image=np.array(image))['image']
                ori_image = self.transform['val'](image=np.array(image))['image']
                return ori_image, aug_image, torch.tensor(y, dtype=torch.long)
            else:
                image = self.transform['val'](image=np.array(image))['image']
                return image, torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.label)

class TestDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.transform = Compose([
            Resize(224,224, p=1.0),
            Normalize(mean=(0.55720607, 0.47626135, 0.44402045), std=(0.2195448 , 0.21720374 ,0.22056428), max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0) 

    def __getitem__(self, index):
        image = Image.open(self.img_path[index])
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_path)

