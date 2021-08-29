import os
import sys
import gzip
import random
import platform
import warnings
import collections
from tqdm import tqdm, tqdm_notebook
import re
import requests
import numpy as np
import time
import copy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_iris
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
train_dir = '/opt/ml/input/data/train'
test_dir = '/opt/ml/input/data/eval'
train_image = train_dir + '/images'
train_image_UTK = train_dir + '/UTKFace_revised'
train_df = pd.read_csv(train_dir + '/train.csv')
test_df = pd.read_csv(test_dir + '/info.csv')
path_df = train_dir + '/train.csv'
path_df_UTK  = train_dir + '/UTKFace_total.csv'
model_name = "i2U3M2U3i2.pth"
Imagesize = (256, 192)
