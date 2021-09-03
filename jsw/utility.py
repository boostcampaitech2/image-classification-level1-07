import random
import numpy as np
from tqdm import tqdm

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