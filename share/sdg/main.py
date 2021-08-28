from train_parallel import train
import os
import pandas as pd
import torch
import timm
from tqdm import tqdm
from dataset import generate_testdataset
import numpy as np
from model import Classification_parallel

train_dir = '/opt/ml/input/data/train'
train_info = pd.read_csv(os.path.join(train_dir, 'train_UTKFace.csv'))
valid_info = pd.read_csv(os.path.join(train_dir, 'train_labeled_val.csv'))
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
test_image_dir = os.path.join(test_dir, 'images')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]



model_name = 'efficientnetv2_rw_s'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=16
early_stop=15
learning_rate=1e-3
num_epochs=50

torch.cuda.empty_cache()
dataloaders_test = generate_testdataset(test_image_paths)

best_state_dict = train(model_name, num_epochs, batch_size, early_stop, learning_rate, train_info, valid_info)
torch.save(best_state_dict, '/opt/ml/input/data/eval/best_param/model.pt')

model = Classification_parallel(model_name, device).to(device)
model.load_state_dict(best_state_dict)
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