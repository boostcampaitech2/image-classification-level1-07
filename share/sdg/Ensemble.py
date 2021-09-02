from train import train_normal
import os
import pandas as pd
import torch
import timm
from tqdm import tqdm
from dataset import TestDataset
import numpy as np
from model_normal import Classification_normal, get_classweight
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn
from albumentations import *
from albumentations.pytorch import ToTensorV2

train_dir = '/opt/ml/input/data/train'
train_df = pd.read_csv(os.path.join(train_dir, 'combine.csv'))
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
test_image_dir = os.path.join(test_dir, 'face_crop_eval')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
best_param_dir = os.path.join(test_dir, 'best_param')
mean_ = [0.485, 0.456, 0.406]
std_ = [0.229, 0.224, 0.225]
model_name = 'vit_large_patch16_224'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloaders_test = DataLoader(TestDataset(test_image_paths), shuffle=False)
TestDataset2 = TestDataset(test_image_paths)
TestDataset2.transform = Compose([
    Resize(width=224,height=224),
    Normalize(mean=mean_, std=std_),
    ToTensorV2(),
])
dataloaders_test2 = DataLoader(TestDataset2, shuffle=False)
def get_cnn_vit():
    model = timm.create_model('vit_large_r50_s32_224', pretrained=True, num_classes = 18)
    nn.init.kaiming_normal_(model.head.weight)
    return model


model_1 = Classification_normal(model_name, device, get_classweight(train_df)).to(device)
model_1.load_state_dict(torch.load(os.path.join(best_param_dir, 'model_vitL_2.pt')))
model_2 = get_cnn_vit().to(device)
model_2.load_state_dict(torch.load(os.path.join(best_param_dir, '5.pth')))

model_1.eval()
model_2.eval()
# model_3.eval()

all_predictions = []
dataloaders_test2_iter = iter(dataloaders_test2)
for images in tqdm(dataloaders_test):
    images2 = dataloaders_test2_iter.next()
    with torch.no_grad():
        images = images.to(device)
        images2 = images2.to(device)
        output1 = model_1(images)
        output2 = model_2(images2)
        output = output1 * 0.55 + output2 * 0.45
        pred = output.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission_1.csv'), index=False)
print('test inference is done!')