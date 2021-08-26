from train import train
import os
import pandas as pd
import torch
import timm
from tqdm import tqdm
from dataset import generate_testdataset
import numpy as np

train_dir = '/opt/ml/input/data/train'
train_info = pd.read_csv(os.path.join(train_dir, 'train_labeled.csv'))
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
test_image_dir = os.path.join(test_dir, 'images')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]

model_name = 'efficientnetv2_rw_s'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=64
early_stop=5
learning_rate=1e-3
num_epochs=5

torch.cuda.empty_cache()

model_age_dict = train(train_info= train_info, 
                       option = 'age', 
                       model_name = model_name, 
                       device = device, 
                       batch_size = batch_size, 
                       early_stop = early_stop, 
                       learning_rate=learning_rate, 
                       num_epochs=num_epochs)
model_gender_dict = train(train_info= train_info, 
                       option = 'gender', 
                       model_name = model_name, 
                       device = device, 
                       batch_size = batch_size, 
                       early_stop = early_stop, 
                       learning_rate=learning_rate, 
                       num_epochs=num_epochs)
model_mask_dict = train(train_info= train_info, 
                       option = 'mask', 
                       model_name = model_name, 
                       device = device, 
                       batch_size = batch_size, 
                       early_stop = early_stop, 
                       learning_rate=learning_rate, 
                       num_epochs=num_epochs)

dataloaders_test = generate_testdataset(test_image_paths)

model_age= timm.create_model(model_name, pretrained=True, num_classes=3).to(device=device)
model_gender= timm.create_model(model_name, pretrained=True, num_classes=2).to(device=device)
model_mask= timm.create_model(model_name, pretrained=True, num_classes=3).to(device=device)

model_age.load_state_dict(model_age_dict)
model_gender.load_state_dict(model_gender_dict)
model_mask.load_state_dict(model_mask_dict)

model_age.eval()
model_gender.eval()
model_mask.eval()



# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
prediction=[]
for images in tqdm(dataloaders_test):
    with torch.no_grad():
        images = images.to(device)
        pred_age = model_age(images)
        pred_gender = model_gender(images)
        pred_mask = model_mask(images)
        for age_i in len(pred_age):
            for gender_i in len(pred_gender):
                for mask_i in len(pred_mask):
                    all_predictions[6*mask_i + 3*gender_i + age_i] = pred_age[age_i] + pred_gender[gender_i] + pred_mask[mask_i]
        output = np.argmax(all_predictions)
        prediction.extend(output.cpu().numpy())

        # _, output = torch.max(pred, 1)
        # all_predictions.extend(output.cpu().numpy())
submission['ans'] = prediction
# # 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')