import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset import generate_dataloader_parallel
from model import Classification_parallel
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

def train(model_name, num_epochs, batch_size, early_stop, learning_rate, train_info, valid_info):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classification_parallel(model_name, device).to(device)
    # classifier_mask = [p for p in model.mask_fc.parameters()]
    # classifier_gender = [p for p in model.gender_fc.parameters()]
    # classifier_age = [p for p in model.age_fc.parameters()]
    # classifier_mask.extend(classifier_gender)
    # classifier_mask.extend(classifier_age)
    # classifier_params_collect = classifier_mask
    classifier_params_collect = [p for p in model.classifier_fc.parameters()]
    classifier_params = [{'params' : classifier_params_collect, 'lr' : learning_rate}]
    optimizer = optim.SGD(classifier_params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    dataloaders_train, dataloaders_valid, class_num = generate_dataloader_parallel(train_info, valid_info, batch_size)

    best_f1 = 0
    early_stop_count = 0
    
    for epoch in range(num_epochs):
        print('*** Epoch {} ***'.format(epoch))

        model.train()
        iter_train_loss = []
        iter_valid_loss = []
        iter_train_acc = []
        iter_valid_acc = []
        iter_valid_f1 = []

        iter_train_mask_loss = []
        iter_train_gender_loss = []
        iter_train_age_loss = []
        iter_valid_mask_loss = []
        iter_valid_gender_loss = []
        iter_valid_age_loss = []
        

        for inputs, labels in tqdm(dataloaders_train):
            inputs = inputs.to(device)
            labels = torch.transpose(labels, 0,1).to(device)
            

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output_loss, loss, pred = model.get_loss(inputs, labels, flag=True)
                loss.backward()
                optimizer.step()

                iter_train_loss.append(loss.cpu().item())
                iter_train_acc.extend((pred == labels[3]).cpu().tolist())
                iter_train_mask_loss.append(output_loss[0].cpu().item())
                iter_train_gender_loss.append(output_loss[1].cpu().item())
                iter_train_age_loss.append(output_loss[2].cpu().item())

        model.eval()

        for inputs, labels in tqdm(dataloaders_valid):
            inputs = inputs.to(device)
            labels = torch.transpose(labels, 0, 1).to(device)

            with torch.set_grad_enabled(False):
                
                # statistics
                output_loss, valid_loss, valid_pred = model.get_loss(inputs, labels, flag=False)
                iter_valid_loss.append(valid_loss.cpu().item())
                iter_valid_acc.extend((valid_pred == labels[3]).cpu().tolist())
                iter_valid_mask_loss.append(output_loss[0].cpu().item())
                iter_valid_gender_loss.append(output_loss[1].cpu().item())
                iter_valid_age_loss.append(output_loss[2].cpu().item())
                iter_f1_score = f1_score(y_true=labels[3].cpu().numpy(), y_pred=valid_pred.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_train_mask_loss = np.mean(iter_train_mask_loss)
        epoch_train_gender_loss = np.mean(iter_train_gender_loss)
        epoch_train_age_loss = np.mean(iter_train_age_loss)
        epoch_valid_mask_loss = np.mean(iter_valid_mask_loss)
        epoch_valid_gender_loss = np.mean(iter_valid_gender_loss)
        epoch_valid_age_loss = np.mean(iter_valid_age_loss)
        epoch_train_acc = np.mean(iter_train_acc) * 100
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)
        
        scheduler.step(valid_loss)
        
        print(
                f"[Epoch {epoch}] "
                f"train mask loss : {epoch_train_mask_loss:.4f} | train gender loss : {epoch_train_gender_loss:.4f} | train age loss : {epoch_train_age_loss:.4f} "
                f"train loss : {epoch_train_loss:.4f} | train acc : {epoch_train_acc:.2f}% "
                f"valid mask loss : {epoch_valid_mask_loss:.4f} | valid gender loss : {epoch_valid_gender_loss:.4f} | valid age loss : {epoch_valid_age_loss:.4f} "
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




