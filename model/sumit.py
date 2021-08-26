import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms
# from torchvision.transforms import Resize, ToTensor, Normalize

test_dir = '/opt/ml/input/data/eval'

import cv2
import albumentations as A
from albumentations.augmentations.transforms import CLAHE
from albumentations.pytorch import ToTensorV2

from model import BooDuckMaskModel, GenderNetwork, AgeNetwork, MaskNetwork, device
from data_loader import get_data_transform

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            _image = cv2.imread(self.img_paths[index])
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=_image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)

# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
data_transform = get_data_transform(width=288, height=288, max_size=384)
transform = data_transform['val']
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')
# model = MyModel(num_classes=18).to(device)

TARGET = None
TIMM_MODEL = 'efficientnetv2_rw_s'

model = BooDuckMaskModel(model_name=TIMM_MODEL, person_task=False)
model = model.to(device)
model.load_state_dict(torch.load('backup_effNetV2_FocalLoss/effNetV2_weights.10.pth'))
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')
