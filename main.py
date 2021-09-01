from train import train_normal
import os
import pandas as pd
import torch
import timm
from tqdm import tqdm
from dataset import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from model_normal import Classification_normal, get_classweight
import wandb
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


train_dir = '/opt/ml/input/data/train'
train_df = pd.read_csv(os.path.join(train_dir, 'combine.csv'))
valid_df = pd.read_csv(os.path.join(train_dir, 'train_labeled_val.csv'))
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
test_image_dir = os.path.join(test_dir, 'face_crop_eval')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]



model_name = 'vit_large_patch16_224'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=32
early_stop=4
learning_rate=1e-3
num_epochs=6
cutMix_Flag = False
randomsampler_Flag = False
seed_everything(44)

# config={"epochs": num_epochs, "batch_size": batch_size, "learning_rate" : learning_rate, "cutMix" : cutMix_Flag}
# wandb.init(project = "Image_Classification", entity='donggunseo', config=config)

torch.cuda.empty_cache()
dataloaders_test = DataLoader(TestDataset(test_image_paths), shuffle = False)

best_state_dict = train_normal(model_name, num_epochs, batch_size, early_stop, learning_rate, train_df, valid_df, cutMix=cutMix_Flag, randomsampler=randomsampler_Flag )
torch.save(best_state_dict, '/opt/ml/input/data/eval/best_param/model_vitL_3.pt')

model = Classification_normal(model_name, device, get_classweight(train_df)).to(device)
model.load_state_dict(best_state_dict)
# model.load_state_dict(torch.load('/opt/ml/input/data/eval/best_param/model.pt'))
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
submission.to_csv(os.path.join(test_dir, 'submission_vitL_3.csv'), index=False)
print('test inference is done!')