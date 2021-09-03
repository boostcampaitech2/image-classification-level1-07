import torch
import torchvision
from efficientnet_pytorch import EfficientNet

def get_efficient_b1():
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=18)
    
    return model
    
    
    