import torchvision
import torch.nn as nn
import timm

def build_wideresnet():
    model = torchvision.models.wide_resnet101_2(pretrained=True)
    in_feat = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, 18))
    nn.init.xavier_normal_(model.fc[1].weight)
    print("Model: WideResNet")
    
    return model

def build_hybridViT():
    model = timm.create_model('vit_large_r50_s32_224', pretrained=True, num_classes = 18)
    nn.init.kaiming_normal_(model.head.weight)
    return model