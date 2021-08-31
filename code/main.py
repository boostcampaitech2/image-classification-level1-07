import os
from sys import path
from numpy.lib.npyio import NpzFile
import pandas as pd
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim import optimizer
from torchvision import transforms
from tqdm import tqdm
import argparse
from importlib import import_module

from pytz import timezone
import datetime as dt

from model import get_model, get_model_list
from data import get_transforms, TestDataset, TrainDataset
from trainer import train, k_fold_train, train_cutmix
import wandb

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def create_labels(model, unlabeled_loader):
    tqdm.pandas()
    model.to(device)
    with torch.no_grad():
        model.eval()
        
        list = []
        for imgs, _ in tqdm(unlabeled_loader):
            imgs = imgs.to(device)
            
            labels = model(imgs)
            labels = labels.argmax(1)
            del imgs
            for label in labels:
                list.append(label)
    
    return list

def save_model(best_model_state,optimizer_st,args):
    # 모델 저장
    if args.sudo != True:
        model_save_name =  f'{args.model_name}_{args.epoch_size}_{args.batch_size}_{args.learning_rate}_{args.n_fold}fold_{now}.pth'
    else:
        model_save_name =  f'{args.teacher_model}_{args.t_epoch}_{args.student_model}_{args.s_epoch}_{args.batch_size}_{args.learning_rate}_{now}.pth'
        
    torch.save(best_model_state, os.path.join(model_save_path,model_save_name)) # 모델_epoch_batch_lr_fold
    torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer_st
            },  os.path.join(model_save_path,model_save_name))

    print(f'\n best model state dict : {model_save_name}  is saved!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--random_seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers (default: 2)')
    parser.add_argument('--epoch_size', type=int, default=5, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='TrainDataset', help='dataset type (default: TrainDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=tuple, default=(224,224), help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='model type (default: vit_base_patch16_224)')
    
    # sudo- labeling
    parser.add_argument('--teacher_model', type=str, default='vit_small_patch16_224', help='model type (default: vit_small_patch16_224)')
    parser.add_argument('--student_model', type=str, default='vit_large_patch16_224', help='model type (default: vit_large_patch16_224)')
    parser.add_argument('--t_epoch', type=int, default=4, help='number of epochs to train (default: 1)')
    parser.add_argument('--s_epoch', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--u_dataset', type = str, default = 'UnlabeledDataset')
    parser.add_argument('--com_dataset', type = str, default = 'CombineDataset')
    parser.add_argument('--u_root', type = str, default = '/opt/ml/newinput2')
    
    
    
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 10)')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument('--n_fold', type = int, default = 0, help = 'how many fold using k-fold')
    parser.add_argument('--early_stop', type = int, default = 5, help = 'early stop threshold')
    parser.add_argument('--backbone_freeze', type=float, default=0.5, help = 'backbone freeze ratio')
    parser.add_argument('--beta', type=float, default=1, help = 'cutmix beat : if 0: trian, 1: train_cutmix')

    # path 
    parser.add_argument('--train_dir', type = str, default = '/opt/ml/input/data/train')
    parser.add_argument('--test_dir', type = str, default = '/opt/ml/input/data/eval')
    parser.add_argument('--model_save_path', type=str, default='/opt/ml/model')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/output/givenData')
    # parser.add_argument('--csv_name', type = str, help = 'train csv name')

    args = parser.parse_args()
    print(args)
    
    model_save_path = args.model_save_path
    output_dir = args.output_dir

    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')

    k_fold = args.n_fold > 0
    
    # wandb 
    # 1. Start a new run
    wandb.init(project=f'{args.teacher_model}', entity='yiujin' )
    
    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config = dict (learning_rate = args.learning_rate,
                epochs = args.epoch_size,
                teacher_model = args.teacher_model,
                student_model = args.student_model)
    
    # 3. sweep config
#     sweep_config = {
#     'method': 'grid', #grid, random
#     'metric': {
#         'name': 'loss',
#         'goal': 'minimize'   
#     },
#     'parameters': {
#         'epochs': {
#             'values': [10]
#         },
#         'batch_size': {
#             'values': [64]
#         },
#         'dropout': {
#             'values': [0.3, 0.4, 0.5]
#         },
#         'learning_rate': {
#             'values': [1e-3, 1e-4, 3e-4, 3e-5]
#         },
#         # 'fc_layer_size':{
#         #     'values':[128,256,512]
#         # },
#         'optimizer': {
#             'values': ['adamw', 'sgd']
#         },
#     }
# }
#     sweep_id = wandb.sweep(sweep_config, project=f"{args.model_name}")
    
    # dataset, dataloader 준비
    # csv_name =  'train_total_facecrop.csv'
    # xt, xv, yt, yv = split_dataset(data_path=train_dir, csv_name = csv_name)
    # train_dataset = get_labeled_datasets(paths = xt, labels = yt, train = True)
    # valid_dataset = get_labeled_datasets(paths = xv, labels = yv, train=True)  
    
    # 사람 별로 valid set 나눈 dataset
    tr = '/opt/ml/input/data/train/train_yb.csv'
    vl = '/opt/ml/input/data/train/validation.csv'
    # -- dataset
    dataset_module = getattr(import_module("data"), args.dataset)  # default: TrainDataset
    transform = get_transforms()
    train_dataset = dataset_module(path=tr, transform =transform, train=True)
    valid_dataset = dataset_module(path=vl,  transform = transform, train=False)
    
    train_loader = DataLoader(dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    data_loader = {'train_loader' : train_loader, 'valid_loader' : valid_loader}


    # 모델 학습
    torch.cuda.empty_cache()
    result, best_model_state = None, None
    if not k_fold:
        # result, best_model_state, optimizer_st = train_cutmix(data_loader, args)
        result, best_model_state, optimizer_st = train(data_loader, args, teacher = True)
        
    else:
        kfold_models, result, best_model_state, optimizer_st = k_fold_train(args=args, train_dir=args.train_dir, csv_name='train_revised.csv')

    
    # teacher model을 이용하여 label 생성
    teacher_model = get_model(args.teacher_model)
    teacher_model.load_state_dict(best_model_state)
    
    dataset_module = getattr(import_module("data"), args.u_dataset) # UnlabeledDataset
    uDataset = dataset_module(root = args.u_root, transform =transform, train=True)
    uLoader = DataLoader(dataset=uDataset,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    u_label = create_labels(teacher_model, uLoader)
    
    dataset_module = getattr(import_module("data"), args.com_dataset) # CombineDataset
    combine_dataset = dataset_module(ori_path=tr, u_path =args.u_root, transform =transform, train=True)
    combine_dataset.set_label(u_label)
    
    train_loader = DataLoader(dataset=combine_dataset,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    data_loader = {'train_loader' : train_loader, 'valid_loader' : valid_loader}
    
    # 합친 데이터셋으로 다시 student모델 학습
    student_model = get_model(args.student_model)
    result, best_model_state, optimizer_st = train(data_loader, args,teacher = False)

    save_model(best_model_state, optimizer_st, args)

    # --------------------------------------------------------------------------
    # 모델 저장 및 inference
    # meta 데이터와 이미지 경로를 불러옵니다.
    # print('***************Inference Start***************')
    # submission = pd.read_csv(os.path.join(args.test_dir, 'info.csv'))
    # image_dir = os.path.join(args.test_dir, 'images')
    
    # num_classes = 18

    # # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    # image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    # mean=(0.548, 0.504, 0.479)
    # std=(0.237, 0.247, 0.246)
    # transform = transforms.Compose([
    #         transforms.Resize(args.resize),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean, std=std),
    #     ])
    # test_dataset = TestDataset(image_paths, transform)

    # test_dataloader = DataLoader(dataset=test_dataset,
    #                 batch_size=args.batch_size,
    #                 shuffle=False,
    #                 num_workers=args.num_workers)
    
    # # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    # device = torch.device('cuda')
    # model = get_model(args.model_name).to(device)
    # model.load_state_dict(best_model_state)
    # model.eval()
    # if not k_fold:
    #     # k_fold 사용하지 않았을 경우 inference
    #     all_predictions = []
    #     for images in tqdm(test_dataloader):
    #         with torch.no_grad():
    #             images = images.to(device)
    #             pred = model(images)
    #             pred = pred.argmax(dim=-1)
    #             all_predictions.extend(pred.cpu().numpy())
        
    # else:
    #     # k_fold 학습 사용했을 경우 결과 합쳐서 inference
    #     all_fold_pred = []
    #     for fold_idx, best_model_dict in enumerate(kfold_models,1):
    #         model.load_state_dict(best_model_dict)
    #         model.eval()
            
    #         each_fold_logits = []
            
    #         for iter_idx, img in enumerate(test_dataloader, 1):
    #             with torch.no_grad():
    #                 img = img.to(device)
                    
    #                 pred = model(img)
    #                 each_fold_logits.extend(pred.cpu().tolist())
                    
    #                 print(f'{fold_idx}/{args.n_fold}folds inference iteration {iter_idx}/{len(test_dataloader)} ', end = '\r')
                    
    #         all_fold_pred.append(each_fold_logits)
            
    #     all_fold_pred = np.mean(all_fold_pred, axis = 0)
    #     all_predictions = np.argmax(all_fold_pred, axis = -1)

    # # 제출할 파일을 저장합니다.
    # now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    # submission['ans'] = all_predictions
    # csv_name = f'{args.model_name}_{args.epoch_size}_{args.batch_size}_{args.learning_rate}_{args.n_fold}fold_{now}submission.csv'
    # submission.to_csv(os.path.join(output_dir, csv_name), index=False) # 모델_epoch_batch_lr_#fold
    # print(f'\n test inference csv file : {csv_name} is saved!')
    
    
    