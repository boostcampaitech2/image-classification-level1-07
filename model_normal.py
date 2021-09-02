import torch
import torch.nn as nn
import timm
import torch.optim as optim
import torch.nn.init as init

def get_classweight(train_df):
    return list(train_df['class'].value_counts().sort_index())

class Classification_normal(nn.Module):
    def __init__(self, model_name, device, class_weight):
        super(Classification_normal, self).__init__()
        self.model_name = model_name
        self.device = device
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=18).to(device=self.device)

        self.class_weight = class_weight
        for i in range(len(self.class_weight)):
            self.class_weight[i] = sum(class_weight)/(len(self.class_weight) * self.class_weight[i])
        self.class_weight_final = torch.tensor(self.class_weight).to(device=self.device, dtype=torch.float)

    def forward(self, x):
        output = self.model(x)
        return output

    