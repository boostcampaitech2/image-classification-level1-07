
from moduleinit import *
from dataset import CustomDataset
from dataset import TestDataset
from dataset import MixUpDataset_UTK
from dataset import CustomDataset_UTK
from model import *
from submission import *
from train import *

if __name__ == "__main__":
    torch.cuda.empty_cache()
    p = int(input("train = 1 , submit = 2"))

    if p == 1:
        K = 4
        train_image_dataset(K, 2)
        train_UTK_dataset(K, 3, True)
        train_UTK_dataset_mix(K, 2, True)
        train_UTK_dataset(K, 3, True)
        train_image_dataset(K, 2, True)
        #train dataset, split datase

    elif p == 2:
        submit()
    else:
        print('not commend')
