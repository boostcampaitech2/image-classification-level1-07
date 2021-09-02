from train import train_normal
import os
import pandas as pd
import torch
from tqdm import tqdm
from dataset import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from model_normal import Classification_normal, get_classweight
import wandb
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return 


train_dir = '/opt/ml/input/data/train'
train_df = pd.read_csv(os.path.join(train_dir, 'train_labeled_train.csv'))
valid_df = pd.read_csv(os.path.join(train_dir, 'train_labeled_val.csv'))

model_name = 'vit_large_patch16_224'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=32
early_stop=5
learning_rate=1e-3
num_epochs=12
cutMix_Flag = False
seed_everything(44)
continue_Flag = False
if continue_Flag: 
    continue_dict = torch.load('/opt/ml/input/data/eval/best_param/model_vitL_epoch_1-0.pt')
else:
    continue_dict = None

config={"learning_rate" : learning_rate}
wandb.init(project = "Image_Classification", entity='donggunseo', config=config)

train_normal(model_name, num_epochs, batch_size, early_stop, learning_rate, train_df, valid_df, cutMix=cutMix_Flag, continue_dict = continue_dict)