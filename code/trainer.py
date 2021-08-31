import os
from numpy.core.fromnumeric import argsort
import pandas as pd
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score
from tqdm import tqdm

from model import get_model, get_model_list
from data import split_dataset, get_transforms, TrainDataset
from loss import create_criterion
from importlib import import_module
import wandb

def rand_bbox(size, lam): # size : [Batch_size, Channel, Width, Height]
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  

    # 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)
		
    # 패치 모서리 좌표 값 
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_cutmix(data_loader, args, best_model_state= None):
    print('****************Train start!*****************')
    model_name = args.model_name
    learning_rate = args.learning_rate
    epoch_size = args.epoch_size
    early_stop = args.early_stop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name).to(device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 클래스 별로 이미지 수가 다르기 때문에 imbalance 문제를 완화하기 위해 가장 많은 클래스 이미지 수 / 각 클래스 이미지 수로 나눈 값을 가중치로 사용.
    class_num = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109] # original
    # class_num = [3135, 2529, 609, 4678, 4395, 615, 1338, 1377, 480, 2777, 1446, 258, 943, 893, 281, 1754, 1131, 183] # utk face 
    class_weight = torch.tensor(np.max(class_num) / class_num).to(device=device, dtype=torch.float)
    criterion = create_criterion(args.criterion)
    
    # pretrained 모델을 fine-tuning 하므로 feature map을 추출하는 레이어는 learing rate 0.5 비율만 적용.
    # vit
    feature_extractor = [m for n, m in model.named_parameters() if "head" not in n]
    classifier = [p for p in model.head.parameters()]
    
    #cnn
    # feature_extractor = [m for n, m in model.named_parameters() if "classifier" not in n]
    # classifier = [p for p in model.classifier.parameters()]
    
    params = [
        {"params": feature_extractor, "lr": learning_rate * args.backbone_freeze},
        {"params": classifier, "lr": learning_rate}
    ]
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(params, lr = args.learning_rate)
    # optimizer = AdamW(params, lr=learning_rate)
    
    
    # ConsineAnnealing Scheduler 적용.
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    result = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_f1": [],
    }
    
    train_loader = data_loader["train_loader"]
    valid_loader = data_loader["valid_loader"]
    
    best_model_state = None
    best_f1 = 0
    early_stop_count = 0
    
    wandb.watch(model)
    for epoch_idx in range(1, epoch_size + 1):
        model.train()

        iter_train_loss = []
        iter_valid_loss = []
        iter_valid_acc = []
        iter_valid_f1 = []
        
        for iter_idx, (aug, train_imgs, train_labels) in enumerate(train_loader, 1):
            aug, train_imgs, train_labels = aug.to(device=device, dtype=torch.float),train_imgs.to(device=device, dtype=torch.float), train_labels.to(device)

            optimizer.zero_grad()
            if np.random.random()>0.4: # cutmix가 실행될 경우     
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(train_imgs.size()[0]).to(device)
                target_a = train_labels # 원본 이미지 label
                target_b = train_labels[rand_index] # 패치 이미지 label       
                bbx1, bby1, bbx2, bby2 = rand_bbox(train_imgs.size(), lam)
                train_imgs[:, :, bbx1:bbx2, bby1:bby2] = train_imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (train_imgs.size()[-1] * train_imgs.size()[-2]))
                outputs = model(train_imgs)
                train_loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam) # 패치 이미지와 원본 이미지의 비율에 맞게 loss를 계산을 해주는 부분

            else: # cutmix가 실행되지 않았을 경우
                del train_imgs
                outputs= model(aug) 
                train_loss= criterion(outputs, train_labels)
            
            train_loss.backward()

            optimizer.step()
            iter_train_loss.append(train_loss.cpu().item())

            print(
                f"[Epoch {epoch_idx}/{epoch_size}] model training iteration {iter_idx}/{len(train_loader)}     ",
                end="\r",
            )


        with torch.no_grad():
            for iter_idx, (valid_imgs, valid_labels) in enumerate(valid_loader, 1):
                model.eval()

                valid_imgs, valid_labels = valid_imgs.to(device=device, dtype=torch.float), valid_labels.to(device)

                valid_pred = model(valid_imgs)
                valid_loss = criterion(valid_pred, valid_labels)

                iter_valid_loss.append(valid_loss.cpu().item())

                valid_pred_c = valid_pred.argmax(dim=-1)
                iter_valid_acc.extend((valid_pred_c == valid_labels).cpu().tolist())

                iter_f1_score = f1_score(y_true=valid_labels.cpu().numpy(), y_pred=valid_pred_c.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

                print(
                    f"[Epoch {epoch_idx}/{epoch_size}] model validation iteration {iter_idx}/{len(valid_loader)}     ",
                    end="\r"
                )

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)
        if epoch_idx % args.log_interval == 0:
            wandb.log({"epoch_train_loss": epoch_train_loss, "epoch_valid_loss": epoch_valid_loss,
                    "epoch_valid_acc": epoch_valid_acc, "epoch_valid_f1_score": epoch_valid_f1_score})
            
        
        result["train_loss"].append(epoch_train_loss)        
        result["valid_loss"].append(epoch_valid_loss)        
        result["valid_acc"].append(epoch_valid_acc)        
        result["valid_f1"].append(epoch_valid_f1_score)

        scheduler.step()

        print(
            f"[Epoch {epoch_idx}/{epoch_size}] "
            f"train loss : {epoch_train_loss:.4f} | "
            f"valid loss : {epoch_valid_loss:.4f} | valid acc : {epoch_valid_acc:.2f}% | valid f1 score : {epoch_valid_f1_score:.4f}"
        )

        if epoch_valid_f1_score > best_f1:
            best_f1 = epoch_valid_f1_score
            best_model_state = model.state_dict()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == early_stop:
            print("early stoped." + " " * 30)
            break

    print('**************Training Done!!********************')
    return result, best_model_state,  optimizer.state_dict()



def k_fold_train(args, train_dir='/opt/ml/input/data/train', csv_name = 'train_revised.csv'):

    traindf = pd.read_csv(os.path.join(train_dir, csv_name))
    X = traindf['path']
    y = traindf['class']

    skf = StratifiedKFold(n_splits = args.n_fold, shuffle = True, random_state = args.random_seed)

    kfold_models = []
    best_model_state = None
    print('****************K Fold Train start!*****************')
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        print(f'-----------{fold} fold------------')
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        xt, xv, yt, yv = split_dataset(data_path=train_dir, csv_name = csv_name)
        train_dataset = get_labeled_datasets(paths = xt, labels = yt, train=True)
        valid_dataset = get_labeled_datasets(paths = xv, labels = yv, train=True)  
        
        train_loader = get_loader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers = args.num_workers)
        valid_loader = get_loader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers = args.num_workers)

        data_loader = {'train_loader' : train_loader, 'valid_loader' : valid_loader}

        result, best_model_state = train(data_loader, args, best_model_state)
        kfold_models.append(best_model_state)
        
    print('**************K-Fold Training Done!!********************')
    return kfold_models, result, best_model_state



def train(data_loader, args, teacher,best_model_state= None):
    print('****************Train start!*****************')
    
    learning_rate = args.learning_rate
    if teacher == True:
        model_name = args.teacher_model
        epoch_size = args.t_epoch
    else:
        model_name = args.student_model
        epoch_size = args.s_epoch
    early_stop = args.early_stop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name).to(device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 클래스 별로 이미지 수가 다르기 때문에 imbalance 문제를 완화하기 위해 가장 많은 클래스 이미지 수 / 각 클래스 이미지 수로 나눈 값을 가중치로 사용.
    class_num = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109] # original
    # class_num = [3135, 2529, 609, 4678, 4395, 615, 1338, 1377, 480, 2777, 1446, 258, 943, 893, 281, 1754, 1131, 183] # utk face 
    class_weight = torch.tensor(np.max(class_num) / class_num).to(device=device, dtype=torch.float)
    criterion = create_criterion(args.criterion)

    
    # pretrained 모델을 fine-tuning 하므로 feature map을 추출하는 레이어는 learing rate 0.5 비율만 적용.
    # vit
    feature_extractor = [m for n, m in model.named_parameters() if "head" not in n]
    classifier = [p for p in model.head.parameters()]
    
    #cnn
    # feature_extractor = [m for n, m in model.named_parameters() if "classifier" not in n]
    # classifier = [p for p in model.classifier.parameters()]
    params = [
        {"params": feature_extractor, "lr": learning_rate * args.backbone_freeze},
        {"params": classifier, "lr": learning_rate}
    ]
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(params, lr = args.learning_rate)
    
    
    # Scheduler 적용.
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    result = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_f1": [],
    }
    
    train_loader = data_loader["train_loader"]
    valid_loader = data_loader["valid_loader"]
    
    best_model_state = None
    best_f1 = 0
    early_stop_count = 0

    wandb.watch(model)
    for epoch_idx in range(1, epoch_size + 1):
        model.train()

        iter_train_loss = []
        iter_valid_loss = []
        iter_valid_acc = []
        iter_valid_f1 = []
        
        for iter_idx, (train_imgs, train_labels) in enumerate(train_loader, 1):
            train_imgs, train_labels = train_imgs.to(device=device, dtype=torch.float), train_labels.to(device)

            optimizer.zero_grad()

            train_pred = model(train_imgs)
            train_loss = criterion(train_pred, train_labels)
            train_loss.backward()

            optimizer.step()
            iter_train_loss.append(train_loss.cpu().item())

            print(
                f"[Epoch {epoch_idx}/{epoch_size}] model training iteration {iter_idx}/{len(train_loader)}     ",
                end="\r",
            )


        with torch.no_grad():
            for iter_idx, (valid_imgs, valid_labels) in enumerate(valid_loader, 1):
                model.eval()

                valid_imgs, valid_labels = valid_imgs.to(device=device, dtype=torch.float), valid_labels.to(device)

                valid_pred = model(valid_imgs)
                valid_loss = criterion(valid_pred, valid_labels)

                iter_valid_loss.append(valid_loss.cpu().item())

                valid_pred_c = valid_pred.argmax(dim=-1)
                iter_valid_acc.extend((valid_pred_c == valid_labels).cpu().tolist())

                iter_f1_score = f1_score(y_true=valid_labels.cpu().numpy(), y_pred=valid_pred_c.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

                print(
                    f"[Epoch {epoch_idx}/{epoch_size}] model validation iteration {iter_idx}/{len(valid_loader)}     ",
                    end="\r"
                )

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)
        if epoch_idx % args.log_interval == 0:
            wandb.log({"epoch_train_loss": epoch_train_loss, "epoch_valid_loss": epoch_valid_loss,
                    "epoch_valid_acc": epoch_valid_acc, "epoch_valid_f1_score": epoch_valid_f1_score})
        
        result["train_loss"].append(epoch_train_loss)        
        result["valid_loss"].append(epoch_valid_loss)        
        result["valid_acc"].append(epoch_valid_acc)        
        result["valid_f1"].append(epoch_valid_f1_score)

        scheduler.step()

        print(
            f"[Epoch {epoch_idx}/{epoch_size}] "
            f"train loss : {epoch_train_loss:.4f} | "
            f"valid loss : {epoch_valid_loss:.4f} | valid acc : {epoch_valid_acc:.2f}% | valid f1 score : {epoch_valid_f1_score:.4f}"
        )

        if epoch_valid_f1_score > best_f1:
            best_f1 = epoch_valid_f1_score
            best_model_state = model.state_dict()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == early_stop:
            print("early stoped." + " " * 30)
            break

    print('**************Training Done!!********************')
    return result, best_model_state,  optimizer.state_dict()

    
