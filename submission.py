import os
import argparse
from random import shuffle
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from albumentations import *
from albumentations.pytorch.transforms import *
from torch.utils.data import Dataset, DataLoader

from share_sdg.sdg.model_normal import *
from share_jsw.model import *

parser = argparse.ArgumentParser()

parser.add_argument('--model_a', type=str, required=True, help='model a pth file path')
parser.add_argument('--model_b', type=str, required=True, help='model b pth file path')
parser.add_argument('--root', type=str, required=True, help='test data root path')


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    args = parser.parse_args()
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print('running in cuda')
    else:
        device = torch.device("cpu")
        print('running in cpu')
    
    model_name = 'vit_large_patch16_224'
    model_b = Classification_normal(model_name, device)
    model_b.load_state_dict(torch.load(args.mode_b)['model_state_dict'])
    model_b = model_b.to(device)
    
    model_a = build_hybridViT()
    model_a.load_state_dict(torch.load(args.model_a))
    model_a = model_a.to(device)
    
    mean_a = [0.485, 0.456, 0.406]
    std_a = [0.229, 0.224, 0.225]
    mean_b = [0.55720607, 0.47626135, 0.44402045]
    std_b = [0.2195448 , 0.21720374 ,0.22056428]
    
    transforms_a = Compose(
        Resize(height=224, width=224, always_apply=True),
        Normalize(mean_a, std_a),
        ToTensorV2()
    )
    
    transforms_b = Compose(
        Resize(height=224, width=224, always_apply=True),
        Normalize(mean_b, std_b),
        ToTensorV2()
    )
    
    submission = pd.read_csv(os.path.join(args.root, 'info.csv'))
    image_dir = os.path.join(args.root, 'images')
    
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset_a = TestDataset(image_paths, transforms_a)
    dataset_b = TestDataset(image_paths, transforms_b)
    
    loader_a = DataLoader(dataset_a, shuffle=False)
    loader_b = DataLoader(dataset_b, shuffle=False)
    
    loader_b_iter = iter(loader_b)
    
    all_predictions = []
    for images in tqdm(loader_a):
        images_b = loader_b_iter.next()
        with torch.no_grad():
            images = images.to(device)
            images_b = images_b.to(device)
            pred_a = model_a(images)
            pred_b = model_b(images)
            
            pred = pred_a + pred_b
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions
    
    submission.to_csv(os.path.join(args.root, 'submission.csv'), index=False)
    print('test inference is done!')