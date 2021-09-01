import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset import TrainDataset
from model_normal import Classification_normal, get_classweight
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import DataLoader
import wandb
from torch.utils.data.sampler import WeightedRandomSampler

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

def get_weighted_random_sampler(class_cnts, dataset):
    tqdm.pandas()
    #? count all class numbers
    class_weights = [1./c for c in class_cnts]
    class_weights = torch.FloatTensor(class_weights)
    #? applying class weights
    print('Applying class weights')
    weights = []
    for idx in tqdm(range(len(dataset))):
        _, _, y = dataset[idx]
        weights.append(class_weights[y])
    # weights = [class_weights[y] for _, _, y in tqdm(dataset) if y!=27907]
    weights = torch.FloatTensor(weights)

    #? build sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def train_normal(model_name, num_epochs, batch_size, early_stop, learning_rate, train_df, valid_df, cutMix=True, randomsampler = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weight = get_classweight(train_df)
    model = Classification_normal(model_name, device, class_weight)
    criterion = nn.CrossEntropyLoss(weight=model.class_weight_final)
    feature_extractor = [m for n, m in model.named_parameters() if "head" not in n]
    classifier = [p for p in model.model.head.parameters()]
    params = [
                    {"params": feature_extractor, "lr": learning_rate * 0.01},
                    {"params": classifier, "lr": learning_rate}
    ]
    optimizer = optim.AdamW(params, lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    if randomsampler:
        sampler = get_weighted_random_sampler(class_weight, TrainDataset(train_df, train=True))
        dataloaders_train = DataLoader(TrainDataset(train_df, train=True), batch_size= batch_size, sampler=sampler, num_workers = 4)
    else:
        dataloaders_train = DataLoader(TrainDataset(train_df, train=True), batch_size= batch_size, shuffle=True, num_workers = 4)
    dataloaders_valid = DataLoader(TrainDataset(valid_df, train=False), batch_size= batch_size, shuffle = False, num_workers = 4)

    best_f1 = 0
    early_stop_count = 0
    
    for epoch in range(num_epochs):
        print('*** Epoch {} ***'.format(epoch))

        # train
        model.train()
        iter_train_loss = []
        iter_valid_loss = []
        iter_train_acc = []
        iter_valid_acc = []
        iter_valid_f1 = []

        for inputs, inputs_aug, labels in tqdm(dataloaders_train):
            inputs = inputs.to(device)
            inputs_aug = inputs_aug.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                if cutMix:
                    if np.random.random()>0.4:
                        lam = np.random.beta(1.0, 1.0)
                        rand_index = torch.randperm(inputs.size()[0]).to(device)
                        target_a = labels # 원본 이미지 label
                        target_b = labels[rand_index] # 패치 이미지 label       
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        output = model(inputs)
                        loss = criterion(output, target_a)*lam + criterion(output,target_b)*(1.-lam)
                    else:
                        del inputs
                        output = model(inputs_aug)
                        loss = criterion(output, labels)
                else:
                    output = model(inputs_aug)
                    loss = criterion(output, labels)
                _, pred = torch.max(output, 1)
                loss.backward()
                optimizer.step()

                iter_train_loss.append(loss.cpu().item())
                iter_train_acc.extend((pred == labels).cpu().tolist())
    
        model.eval()
        #valid
        for inputs, labels in tqdm(dataloaders_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                output = model(inputs)
                valid_loss = criterion(output, labels)
                _, valid_pred = torch.max(output, 1)
                iter_valid_loss.append(valid_loss.cpu().item())
                iter_valid_acc.extend((valid_pred == labels).cpu().tolist())
                iter_f1_score = f1_score(y_true=labels.cpu().numpy(), y_pred=valid_pred.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

        
        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_train_acc = np.mean(iter_train_acc) * 100
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)
        # wandb.log({'train_accuracy' : epoch_train_acc, 'train_loss' : epoch_train_loss, 'valid_accuracy' : epoch_valid_acc, 'valid_loss' : epoch_valid_loss, 'valid_f1_score' : epoch_valid_f1_score})

        # scheduler.step(valid_loss)

        print(
                    f"[Epoch {epoch}] \n"
                    f"train loss : {epoch_train_loss:.4f} | train acc : {epoch_train_acc:.2f}% \n"
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
        
    return best_model_state












    