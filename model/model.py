import os
import time
from datetime import datetime
import copy
import logging

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# from torchvision.models import resnet18
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

now = datetime.now()
cur_time_str = now.strftime("%d%m%Y_%H%M")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-8s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=f'./{cur_time_str}.log',
                    filemode='w')

logger = logging.getLogger('model')


# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False

# def initialize_resnet18_model(num_classes, feature_extract=True, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.

#     model_ft = resnet18(pretrained=use_pretrained)
#     set_parameter_requires_grad(model_ft, feature_extract)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, num_classes)
#     input_size = 224

#     return model_ft, input_size

class AgeNetwork(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(AgeNetwork, self).__init__()
        # self.resnet18, _ = initialize_resnet18_model(3, False)
        self.timm_pretrained = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        # return self.resnet18(x)
        return self.timm_pretrained(x)

class GenderNetwork(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(GenderNetwork, self).__init__()
        # self.resnet18, _ = initialize_resnet18_model(2, False)
        self.timm_pretrained = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        # return self.resnet18(x)
        return self.timm_pretrained(x)

class PersonNetwork(nn.Module):
    def __init__(self, model_name, num_classes=6):
        super(PersonNetwork, self).__init__()
        self.timm_pretrained = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.timm_pretrained(x)

class MaskNetwork(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(MaskNetwork, self).__init__()
        # self.resnet18, _ = initialize_resnet18_model(3, False)
        # self.efficientnetv2_rw_s = timm.create_model(model_name=model_name, num_classes=3, pretrained=True)
        self.timm_pretrained = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        # return self.resnet18(x)
        return self.timm_pretrained(x)
        # return self.efficientnetv2_rw_s(x)


class FocalLoss(nn.Module):
    """
    https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    """
    def __init__(self, gamma=2., reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.nll_loss = nn.NLLLoss(reduction=self.reduction)
        self.nll_loss = nn.NLLLoss()
        
    def forward(self, input_tensor, target_tensor):
        log_prob = self.log_softmax(input_tensor)
        prob = torch.exp(log_prob)
        return self.nll_loss(((1-prob)**self.gamma)*log_prob, target_tensor)


class BooDuckMaskModel(nn.Module):
    def __init__(self, model_name, person_task=False):
        super(BooDuckMaskModel, self).__init__()
        ### Init var
        self.person_task = person_task

        ### Each Model's Parms
        if self.person_task:
            self.person_network = PersonNetwork(model_name)
            self.person_loss_fn = FocalLoss()
        else:
            self.age_network = AgeNetwork(model_name)
            self.gender_network = GenderNetwork(model_name)
            self.age_loss_fn = FocalLoss()
            self.gender_loss_fn = FocalLoss()

        self.mask_network = MaskNetwork(model_name)
        self.mask_loss_fn = FocalLoss()

        ### BooDuck Parms
        if self.person_task:
            self.person_fc = nn.Linear(6, 18)
        else:
            self.age_fc = nn.Linear(3, 18)
            self.gender_fc = nn.Linear(2, 18)
        self.mask_fc = nn.Linear(3, 18)

        self.label_loss_fn = FocalLoss()

        ### ETC
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def get_target_loss_and_stat(self, x, y):
        # return self.age_loss_fn(x, y)
        # return self.gender_loss_fn(x, y)
        return self.mask_loss_fn(x, y)

    def get_loss_and_stat(self, x, y, phase):

        if self.person_task:
            person_outputs = self.person_network(x) # Infer
            person_loss = self.person_loss_fn(person_outputs, y[:, -3]) # Loss
        else:
            # Infer
            age_outputs = self.age_network(x)
            gender_outputs = self.gender_network(x)
            # Loss
            age_loss =  self.age_loss_fn(age_outputs, y[:, -4])
            gender_loss = self.gender_loss_fn(gender_outputs, y[:, -3])

        mask_outputs = self.mask_network(x) # Infer
        mask_loss = self.mask_loss_fn(mask_outputs, y[:, -2]) # Loss

        if phase == 'train':  # Backward
            mask_loss.backward()
            if self.person_task:
                person_loss.backward()
            else:
                age_loss.backward()
                gender_loss.backward()

        if self.person_task:  # Get logits
            label_logits = self.person_fc(person_outputs.detach())
        else:
            label_logits = self.age_fc(age_outputs.detach())
            label_logits += self.gender_fc(gender_outputs.detach())
        label_logits += self.mask_fc(mask_outputs.detach())

        # Label loss and pred.
        label_prob = self.log_softmax(label_logits)
        label_loss = self.label_loss_fn(label_prob, y[:, -1])
        _, preds = torch.max(label_prob, 1)

        if self.person_task:  # Get logits
            _stat = {
                'person_loss': person_loss.cpu().detach().numpy(),
                'mask_loss': mask_loss.cpu().detach().numpy(),
            }
        else:
            _stat = {
                'age_loss': age_loss.cpu().detach().numpy(),
                'gender_loss': gender_loss.cpu().detach().numpy(),
                'mask_loss': mask_loss.cpu().detach().numpy(),
                # 'label_prob': label_prob.cpu().detach().numpy(),
                # 'preds': preds,
            }

        return label_loss, preds, _stat

    def train_model(self, dataloaders, optimizer, scheduler=None, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1:>2}/{num_epochs} ----------")
            print(f"Epoch {epoch + 1:>2}/{num_epochs} ----------")

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()  # Set model to training mode
                else:
                    self.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_f1 = 0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                # for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Indivisual train
                        # age_outputs = self.age_network(inputs)
                        # gender_outputs = self.gender_network(inputs)
                        # mask_outputs = self.mask_network(inputs)
                        # _, preds = torch.max(mask_outputs, 1)
                        # loss = self.get_target_loss_and_stat(mask_outputs, labels)

                        # Whole
                        loss, preds, stat = self.get_loss_and_stat(inputs, labels, phase)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels[:, -1].data).detach().item()
                    running_f1 += f1_score(preds.cpu().detach().numpy(), labels[:,-1].cpu().detach().numpy(), average='macro')
                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / dataloaders[phase].dataset.__len__()
                # epoch_acc = running_corrects.double() / dataloaders[phase].dataset.__len__()
                epoch_acc = running_corrects / dataloaders[phase].dataset.__len__()
                epoch_f1 = running_f1 / dataloaders[phase].__len__()

                if self.person_task:
                    logger.info(f"{phase:5} Loss: ({epoch_loss:.4f}) person: {stat['person_loss']:.4f} mask: {stat['mask_loss']:.4f} // Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
                else:
                    logger.info(f"{phase:5} Loss: ({epoch_loss:.4f}) age:{stat['age_loss']:.4f} gender: {stat['gender_loss']:.4f} mask: {stat['mask_loss']:.4f} // Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())
            if ((epoch+1)%5) == 0:
                torch.save(self.state_dict(), f'effNetV2_weights.{epoch+1:02}.pth')

            logger.info(f"")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        torch.save(best_model_wts, 'effNetV2_weights.best.pth')

        # load best model weights
        # self.gender_network.load_state_dict(best_model_wts)
        return self.gender_network

    def save(self,):
        torch.save(self.state_dict(), 'effNetV2_weights.final.pth')

    def forward(self, x):
        if self.person_task:
            person_outputs = self.person_network(x) # Infer
        else:
            # Infer
            age_outputs = self.age_network(x)
            gender_outputs = self.gender_network(x)

        mask_outputs = self.mask_network(x) # Infer

        if self.person_task:  # Get logits
            label_logits = self.person_fc(person_outputs.detach())
        else:
            label_logits = self.age_fc(age_outputs.detach())
            label_logits += self.gender_fc(gender_outputs.detach())
        label_logits += self.mask_fc(mask_outputs.detach())

        return self.log_softmax(label_logits)
