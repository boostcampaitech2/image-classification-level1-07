import torch
import random
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import torch.nn as nn


def train_runner(model, criterion, train_loader, device, val_loader=None, u_loader=None, epochs=10):
    model = model.to(device)
    params = [
        {'params': model._blocks.parameters(), 'lr': 0.0005},
        {'params': model._fc.parameters(), 'lr': 0.005}
    ]
    optimizer = optim.Adam(params, lr=0.0005)
    criterion = criterion.to(device)
    min_loss = -1
    min_cnt = 0
    tqdm.pandas()
    for epoch in range(epochs):
        u_train_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_size = 0
        val_size = 0
        
        model.train()
        
        for imgs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out.to(device), labels)
            
            loss.backward()
            optimizer.step()
            train_size += imgs.shape[0]
            train_loss += loss.item() * imgs.shape[0]
            
            ret, predictions = torch.max(out.data, 1)
            correct_conuts = predictions.eq(labels.data.view_as(predictions))
            
            acc = torch.mean(correct_conuts.type(torch.FloatTensor))
            
            train_acc += acc.item()
        
        if val_loader is not None:
            with torch.no_grad():
                model.eval()
                
                for imgs, labels in tqdm(val_loader):
                    imgs, labels = imgs.to(device), labels.to(device)

                    out = model(imgs)
                    
                    loss = criterion(out, labels)
                    
                    valid_loss += loss.item() * imgs.shape[0]
                    
                    ret, predictions = torch.max(out.data, 1)
                    correct_conuts = predictions.eq(labels.data.view_as(predictions))
                    
                    acc += torch.mean(correct_conuts.type(torch.FloatTensor))
                    
                    valid_acc += acc.item() * imgs.shape[0]
                    val_size += imgs.shape[0]
                    del imgs
                    del labels
        
        
            if min_loss < valid_loss and min_loss != -1:
                min_cnt += 1
            else:
                torch.save(model.state_dict(), '/opt/ml/pseudo_labeling/best_loss.pth')
                min_cnt = 0
                min_loss = valid_loss
            if min_cnt > 3:
                return model
        print(f'Epoch[{epoch}]:')
        print(f'Training Loss {train_loss/train_size:.4f}, Training Accuracy {train_acc/train_size:.4f}')
        if val_loader is not None:
            print(f'Validation Loss {valid_loss/val_size:.4f}, Validation Accuracy: {valid_acc/val_size:.4f}\n\n')
        
    return model

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
    
def set_seed():
    random.seed(33)
    np.random.seed(33)
    torch.manual_seed(33)
    torch.cuda.manual_seed_all(33)
    
