import torch
import torch.nn as nn
import timm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_model(model_name, num_classes, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device=device)
    return model