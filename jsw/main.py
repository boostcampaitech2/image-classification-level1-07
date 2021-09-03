import random
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from data import *
from models import get_efficient_b1
from utility import *


class_nums = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109]
class_weights = [1./c for c in class_nums]
class_weights = torch.FloatTensor(class_weights)

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print('running in cuda')
else:
    device = torch.device("cpu")
    print('running in cpu')
        
def create_labels(model, unlabeled_loader):
    tqdm.pandas()
    model.to(device)
    with torch.no_grad():
        model.eval()
        
        list = []
        for imgs, _ in tqdm(unlabeled_loader):
            imgs = imgs.to(device)
            
            labels = model(imgs)
            labels = labels.argmax(1)
            del imgs
            for label in labels:
                list.append(label)
    
    return list
            

if __name__ == '__main__':
    set_seed()
    labeled_root = '/opt/ml/input/data/train'
    unlabeled_root = '/opt/ml/out_data'
    
    criterion = SmoothCrossEntropy(alpha=0.15)
    model = get_efficient_b1()
    
    labeled_dataset = get_labeled_datasets(labeled_root, 'train_revised.csv', train=True)
    unlabeled_dataset = get_unlabeled_datasetes(unlabeled_root, train=False)
    
    train_len = int(len(labeled_dataset) * 0.85)
    val_len = len(labeled_dataset) - train_len
    
    l_train_set, l_val_set = torch.utils.data.random_split(labeled_dataset, [train_len, val_len])
    
    l_train_loader, l_val_loader = get_train_val_loader(l_train_set, l_val_set)
    
    u_loader = get_loader(unlabeled_dataset)
    
    teacher_model = train_runner(model, criterion, l_train_loader, device, l_val_loader, epochs=30)
    teacher_model.load_state_dict(torch.load('/opt/ml/pseudo_labeling/best_loss.pth'))
    #model.load_state_dict(torch.load('/opt/ml/pseudo_labeling/best_loss.pth'))
    
    
    labels = create_labels(model, u_loader)
    
    del u_loader, model
    
    combined_dataset = CombinedDataset(labeled_root, 'train_revised.csv', unlabeled_root, train=True)
    
    for i, label in enumerate(labels):
        combined_dataset.set_unlabeled_label(i, label)
    
    combined_loader = get_loader(combined_dataset)
    
    student_model = get_efficient_b1()
    final_model = train_runner(student_model, criterion, combined_loader, device, epochs=8)
    
    torch.save(final_model.state_dict(), '/opt/ml/pseudo_labeling/final_model.pth')