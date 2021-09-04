import os
import cv2
import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

#! Referenced: https://github.com/kekmodel/MPL-pytorch/blob/main/utils.py
class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha
        
    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def set_seed(seed=33):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def get_weighted_random_sampler(num_classes, dataset, cutmix=False):
    tqdm.pandas()
    #? count all class numbers
    print('Counting class numbers')
    label_list = []
    class_cnts = [0 for _ in range(num_classes)]
    
    if cutmix:
        for _, _, label in tqdm(dataset):
            class_cnts[label] += 1
            label_list.append(label)
    else:
        for _, label in tqdm(dataset):
            class_cnts[label] += 1
            label_list.append(label)
            
    #? calculate class weights
    class_weights = [1./c for c in class_cnts]
    class_weights = torch.FloatTensor(class_weights)
    
    #? apply class weights
    print('Applying class weights')
    weights = [class_weights[y] for y in tqdm(label_list)]
    weights = torch.FloatTensor(weights)
    
    #? build sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

#! CUTMIX bbox calculation, referenced 최석민_T2221, https://stages.ai/competitions/74/discussion/talk/post/493
def rand_bbox(size, lam): #? size : [Batch_size, Channel, Width, Height]
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  
    #? 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)
	
    #? 패치 모서리 좌표 값 
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def get_optimizer(model, lr1 = 0.05, lr2 = 0.0002, target = None):
    parameter_list = []
    for n, p in model.named_parameters():
        dic = {}
        for t in target:
            if t in n:
                lr = lr1
            else:
                lr = lr2
        dic['params'] = p
        dic['lr'] = lr
        parameter_list.append(dic)
        
    optimizer = optim.SGD(parameter_list, 0.001, momentum=0.9)
    return optimizer

def create_labels(model, loader, num_classes=18, device=None):
    tqdm.pandas()
    if device is None:
        device = torch.device('cpu')
    model = model.to(device)
        
    print('Creating Pseudo Labels')
    with torch.no_grad():
        model.eval()
        list = []
        for imgs, ys in tqdm(loader):
            imgs = imgs.to(device)
            labels = model(imgs)
            labels = labels.argmax(1)
            del imgs
            for label in labels:
                list.append(label)
        return list
    
def face_crop(root, device):
    mtcnn = MTCNN(keep_all=True, device=device)
    root = os.path.join(root, 'images')
    
    for imgs in os.listdir(root):
        if imgs[0] == '.': continue

            
        img_dir = os.path.join(root, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
        
        # boxes 확인
        if len(probs) > 1: 
            print(boxes)
        if not isinstance(boxes, np.ndarray):
            print('Nope!')
            # 직접 crop
            img=img[100:400, 50:350, :]
        
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :]
        
        plt.imsave(os.path.join(root, imgs), img)
        
        
def revise_csv(root):
    u_class = []
    total_list = []
    val_list = []
    tar_id_list = []
    mask = ['1', '2', '3', '4', '5']
    flag = False
    train_df = pd.read_csv(os.path.join(root, 'train.csv'))
    dir = root + '/images'
    for idx in range(len(train_df['id'])):
        id, gender, race, age, path = train_df.iloc[idx]
        
        root = os.path.join(dir, path)
        dirpath, dirnames, filenames = next(os.walk(root))
        for name in filenames:
            if '.' != name[0]:
                id_list = []
                id_list.append(os.path.join(root, name))
                if name[4] in mask:
                    if gender == 'male':
                        if age < 30:
                            age_ = 20
                            id_list.append(0)
                        elif age >= 60:
                            age_ = 60
                            id_list.append(2)
                        else:
                            age_ = 40
                            id_list.append(1)
                    elif gender == 'female':
                        if age < 30:
                            age_ = 20
                            id_list.append(3)
                        elif age >= 60:
                            id_list.append(5)
                            age_ = 60
                        else:
                            age_ = 40
                            id_list.append(4)
                elif name[0] == 'i':
                    if gender == 'male':
                        if age < 30:
                            age_ = 20
                            id_list.append(6)
                        elif age >= 60:
                            age_ = 60
                            id_list.append(8)
                        else:
                            age_ = 40
                            id_list.append(7)
                    elif gender == 'female':
                        if age < 30:
                            age_ = 20
                            id_list.append(9)
                        elif age >= 60:
                            age_ = 60
                            id_list.append(11)
                        else:
                            age_ = 40
                            id_list.append(10)
                elif name[0] == 'n':
                    if gender == 'male':
                        if age < 30:
                            age_ = 20
                            id_list.append(12)
                        elif age >= 60:
                            age_ = 60
                            id_list.append(14)
                        else:
                            age_ = 40
                            id_list.append(13)
                    elif gender == 'female':
                        if age < 30:
                            age_ = 20
                            id_list.append(15)
                        elif age >= 60:
                            age_ = 60
                            id_list.append(17)
                        else:
                            age_ = 40
                            id_list.append(16)
                            
                if id in tar_id_list:
                    flag = True
                if (age_, gender) not in u_class:
                    tar_id_list.append(id)
                    print(age_, gender)
                    u_class.append((age_, gender))
                    flag = True
                elif u_class.count((age_, gender)) < 5 and id not in tar_id_list:
                    tar_id_list.append(id)
                    print(age_, gender)
                    u_class.append((age_, gender))
                    flag = True
                    
                if flag:
                    val_list.append(id_list)
                else:
                    total_list.append(id_list)
                flag = False
    
    header = ['img_path, class']
    with open('./train.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_list)
    with open('./val.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_list)
        
        
#? specific preprocess code for https://www.kaggle.com/tapakah68/medical-masks-p4 dataset
#? crop face if face detected if not, remove image
def preprocess_kaggle_dataset(path, device):
    mtcnn = MTCNN(keep_all=True, device=device)
    idx_list = []
    csv_list = []
    root = os.path.join(path, 'images')
    new_path = os.path.join(path, 'images')
    df = pd.read_csv(os.path.join(path, 'df_part_4.csv'))
    for i in range(len(df['GENDER'])):
        per_list = []
        flag = False
        if not 10 < int(df['AGE'].iloc[i]) < 100:
            idx_list.append(i)
            flag = True
        elif df['GENDER'].iloc[i] != 'MALE' and df['GENDER'].iloc[i] != 'FEMALE':
            idx_list.append(i)
            flag = True
        img_path = os.path.join(root, df['name'].iloc[i])
        if flag:
            if os.path.exists(img_path):
                os.remove(os.path.join(root, df['name'].iloc[i]))
        else:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                boxes,probs = mtcnn.detect(img)
                if not isinstance(boxes, np.ndarray):
                    idx_list.append(i)
                    os.remove(img_path)
                else:
                    xmin = int(boxes[0, 0])-30
                    ymin = int(boxes[0, 1])-30
                    xmax = int(boxes[0, 2])+30
                    ymax = int(boxes[0, 3])+30
                    
                    if xmin < 0: xmin = 0
                    if ymin < 0: ymin = 0
                    if xmax > img.shape[0]: xmax = img.shape[0]
                    if ymax > img.shape[1]: ymax = img.shape[1]
                    
                    img = img[ymin:ymax, xmin:xmax, :]
                    
                    try:
                        img = cv2.resize(img, (224,224))
                        plt.imsave(os.path.join(new_path, df['name'].iloc[i]), img)
                        per_list.append(os.path.join(new_path, df['name'].iloc[i]))
                        if int(df['AGE']) < 30:
                            if int(df['TYPE']) == 1:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(3)
                                else:
                                    per_list.append(0)
                            elif int(df['TYPE']) == 2:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(9)
                                else:
                                    per_list.append(6)
                            else:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(15)
                                else:
                                    per_list.append(12)
                        elif 30 <= int(df['AGE']) < 60:
                            if int(df['TYPE']) == 1:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(4)
                                else:
                                    per_list.append(1)
                            elif int(df['TYPE']) == 2:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(10)
                                else:
                                    per_list.append(7)
                            else:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(16)
                                else:
                                    per_list.append(13)
                        else:
                            if int(df['TYPE']) == 1:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(5)
                                else:
                                    per_list.append(2)
                            elif int(df['TYPE']) == 2:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(11)
                                else:
                                    per_list.append(8)
                            else:
                                if df['GENDER'] == 'FEMALE':
                                    per_list.append(17)
                                else:
                                    per_list.append(14)
                        csv_list.append(per_list)
                    except:
                        print('error')
                    os.remove(os.path.join(img_path))
                    print('done')
            else:
                print('no file')
    header = ['img_path, class']
    with open('./kaggle_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_list)