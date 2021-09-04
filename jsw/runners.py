
import os
import numpy as np
from sklearn.metrics import f1_score

from utility import *
from data import *


def train_cutmix(train_dataset,
        val_dataset,
        model,
        criterion,
        pth_path,
        lr_class,
        lr_back,
        batch_size=64,
        EPOCHS=40,
        BETA=1.0):
    sampler = get_weighted_random_sampler(18, train_dataset, cutmix=True)
    train_loader = get_data_loader(train_dataset, sampler, batch_size=batch_size)
    val_loader = get_data_loader(val_dataset, train=False, batch_size=batch_size)
    smooth_criterion = SmoothCrossEntropy()
    optimizer = get_optimizer(model, lr1=lr_class, lr2=lr_back,vit=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    print('='*100)
    min_loss = -1
    print('Training Begin')
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        f1 = 0.0
        img_list = []

        #? train loop
        model.train()
        for aug, images, labels in tqdm(train_loader):
                model.zero_grad()
                batch_size = images.shape[0]
                images, labels = images.to(device), labels.to(device)
                
                #! CUTMIX applied: referenced 최석민_T2221, https://stages.ai/competitions/74/discussion/talk/post/493
                if np.random.random() > 0.5:
                    tmp = np.random.beta(BETA, BETA)
                    rand_index = torch.randperm(images.size()[0]).to(device)
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), tmp)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    tmp = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    outputs = model(images)
                    loss = criterion(outputs, target_a) * tmp  + criterion(outputs, target_b) * (1.-tmp)
                    del images
                else:
                    del images
                    aug = aug.to(device)
                    outputs = model(aug)
                    loss = smooth_criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_size
                _, pred = torch.max(outputs.data, 1)
                train_acc += (pred == labels).float().mean() * batch_size
                del aug
        with torch.no_grad():
                model.eval()
                for images, labels in tqdm(val_loader):
                    batch_size = images.shape[0]
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                            
                    valid_loss += loss.item() * batch_size
                    _, pred = torch.max(outputs.data, 1)
                    valid_acc += (pred == labels).float().mean() * batch_size
                    
                    f1 += f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
                            
        print(f'Epoch[{epoch+1}]:')
        print(f'Training: Loss = {train_loss/len(train_loader.dataset):.4f}, Acc = {train_acc/len(train_loader.dataset):.4f}')
        print(f'Validation: Loss = {valid_loss/len(val_loader.dataset):.4f}, Acc = {valid_acc/len(val_loader.dataset):.4f}, f1 = {f1/len(val_loader.dataset):.4f}')
        torch.save(model.state_dict(), f'/opt/ml/supervised/checkpoints/{epoch+1}.pth')
        if min_loss == -1 or min_loss >= loss/len(val_loader.dataset):
                torch.save(model.state_dict(), pth_path)
                min_loss = loss/len(val_loader.dataset)
        scheduler.step()

    print('Training Finished')

def train(train_dataset,
        val_dataset,
        model,
        criterion,
        pth_path,
        batch_size=64,
        EPOCHS=30):
    sampler = get_weighted_random_sampler(18, train_dataset)
    train_loader = get_data_loader(train_dataset,sampler, batch_size=batch_size)
    val_loader = get_data_loader(val_dataset, train=False, batch_size=batch_size)
    
    optimizer = get_optimizer(model, lr1=0.05, lr2=0.0001, vit=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    print('='*100)
    max_f1 = -1
    print('Training Begin')
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        f1 = 0.0
        
        #? train loop
        model.train()
        for images, labels in tqdm(train_loader):
                model.zero_grad()
                batch_size = images.shape[0]
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_size
                _, pred = torch.max(outputs.data, 1)
                train_acc += (pred == labels).float().mean() * batch_size
                
        with torch.no_grad():
                model.eval()
                for images, labels in tqdm(val_loader):
                    batch_size = images.shape[0]
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                            
                    valid_loss += loss.item() * batch_size
                    _, pred = torch.max(outputs.data, 1)
                    valid_acc += (pred == labels).float().mean() * batch_size
                    
                    f1 += f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
                            
        print(f'Epoch[{epoch+1}]:')
        print(f'Training: Loss = {train_loss/len(train_loader.dataset):.4f}, Acc = {train_acc/len(train_loader.dataset):.4f}')
        print(f'Validation: Loss = {valid_loss/len(val_loader.dataset):.4f}, Acc = {valid_acc/len(val_loader.dataset):.4f}, f1 = {f1/len(val_loader.dataset):.4f}')
        torch.save(model.state_dict(), os.path.join(pth_path, f'{epoch}.pth'))
        scheduler.step()

        if max_f1 == -1 or max_f1 < f1:
                max_f1 = f1
                torch.save(model.state_dict(), os.path.join(pth_path, 'best_f1.pth'))
        
    print('Training Finished')