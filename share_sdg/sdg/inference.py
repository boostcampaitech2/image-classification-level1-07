import os
import pandas as pd
import torch
from tqdm import tqdm
from dataset import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from model_normal import Classification_normal, get_classweight
from train import seed_everything

model_name = 'vit_large_patch16_224'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

train_dir = '/opt/ml/input/data/train'
train_df = pd.read_csv(os.path.join(train_dir, 'combine.csv'))
valid_df = pd.read_csv(os.path.join(train_dir, 'train_labeled_val.csv'))
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
test_image_dir = os.path.join(test_dir, 'face_crop_eval')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
seed_everything(44)
dataloaders_test = DataLoader(TestDataset(test_image_paths), shuffle = False)

model = Classification_normal(model_name, device, get_classweight(train_df)).to(device)

param_dir = '/opt/ml/input/data/eval/best_param/model_vitL_epoch_1.pt'
best_state_dict = torch.load(param_dir)
model.load_state_dict(best_state_dict['model_state_dict'])
model.eval()

all_predictions = []
for images in tqdm(dataloaders_test):
    with torch.no_grad():
        images = images.to(device)
        output = model(images)
        _, pred = torch.max(output, 1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')