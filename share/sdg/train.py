import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset import generate_dataloader
from model import generate_model
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


def train(train_info, option, model_name, device, batch_size = 32, early_stop = 2, learning_rate=1e-3, num_epochs = 5):
    dataloaders_train, dataloaders_valid, num_classes = generate_dataloader(train_info = train_info, option = option, batch_size=batch_size)
    model = generate_model(model_name=model_name, num_classes = num_classes, device = device)
    early_stop = early_stop
    learning_rate = learning_rate
    num_epochs = num_epochs
    device = device
    best_model_state = None

    if option=='normal':
        class_num = [2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 732, 817, 109, 549, 410, 83, 732, 817, 109]
    elif option=='mask':
        class_num = [2700, 13500, 2700]
    elif option=='gender':
        class_num = [(1042*7),(1658*7)]
    elif option=='age':
         class_num = [(1281 * 7),(1227 * 7),(192 * 7)]
    for i in range(len(class_num)):
        class_num[i] = 18900/(num_classes * class_num[i])
    class_weight = torch.tensor(class_num).to(device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    best_f1 = 0
    early_stop_count = 0

    for epoch in range(num_epochs):
        print('*** Epoch {} ***'.format(epoch))
    
        iter_train_loss = []
        iter_valid_loss = []
        iter_train_acc = []
        iter_valid_acc = []
        iter_valid_f1 = []

        # Training
        model.train()  
            
        for idx, (inputs, labels) in tqdm(enumerate(dataloaders_train)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs= model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # statistics
                iter_train_loss.append(loss.cpu().item())
                train_pred_c = outputs.argmax(dim=-1)
                iter_train_acc.extend((train_pred_c == labels).cpu().tolist())

        # Validation
        model.eval()  
        
        for idx, (inputs, labels) in tqdm(enumerate(dataloaders_valid)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                # statistics
                valid_loss = criterion(outputs, labels)
                iter_valid_loss.append(valid_loss.cpu().item())
                valid_pred_c = outputs.argmax(dim=-1)
                iter_valid_acc.extend((valid_pred_c == labels).cpu().tolist())
                iter_f1_score = f1_score(y_true=labels.cpu().numpy(), y_pred=valid_pred_c.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

        # statistics
        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_train_acc = np.mean(iter_train_acc) * 100
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)
        
        scheduler.step()
        
        print(
                f"[Epoch {epoch}] "
                f"train loss : {epoch_train_loss:.4f} | train acc : {epoch_train_acc:.2f}% "
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






    