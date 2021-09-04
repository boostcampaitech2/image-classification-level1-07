import argparse
from supervised.runners import train_cutmix

import torch.nn as nn

from data import *
from model import *
from utility import *
from runners import *

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, required=True, help='SELECT MODE: PREPROCESS, TRAIN')
parser.add_argument('--data_root', type=str, default='', help='path of data root')
parser.add_argument('--train_csv_file', type=str, default='train.csv', help='train csv file name')
parser.add_argument('--val_csv_file', type=str, default='val.csv', help='val csv file name')
parser.add_argument('--extra_data_root', type=str, default='', help='path of extra data root for pseudo labeling')

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed()
    
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print('running in cuda')
    else:
        device = torch.device("cpu")
        print('running in cpu')
        
    if args.mode == 'TRAIN':
        train_dataset = get_dataset(args.data_root, args.train_csv_file, cutmix=True)
        val_dataset = get_dataset(args.data_root, args.val_csv_file, train=False, cutmix=True)
        
        pth_path = os.path.join(args.data_root, 'checkpoint')
        if not os.path.exists(pth_path):
            os.makedirs(pth_path)
        
        #? train teacher model
        teacher_model = build_wideresnet()
        criterion = nn.CrossEntropyLoss()
        train_cutmix(train_dataset, val_dataset, teacher_model, criterion, lr_class=0.05, lr_back=0.0002, batch_size=128)
        
        del train_dataset
        #? create pseudo label
        teacher_model.load_state_dict(torch.load(os.path.join(pth_path, 'best_loss.pth')))
        unlabeled_dataset = get_unlabeled_dataset(args.extra_data_root, train=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, shuffle=False, batch_size=128, num_workers=4)
        
        labels = create_labels(teacher_model, unlabeled_loader)
        del unlabeled_loader, teacher_model
        
        #? train student model
        criterion = SmoothCrossEntropy(alpha=0.15)
        combined_dataset = get_combined_dataset(args.data_root, args.extra_data_root, args.train_csv_file)
        combined_dataset.set_labels(labels)
        
        model = build_hybridViT()
        train(combined_dataset, val_dataset, model, criterion, batch_size=128)
        
        criterion = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(os.path.join(pth_path, 'best_fl.pth')))
        #? finetune student model
        train_cutmix(train_dataset, val_dataset, model, criterion, lr_class=0.02, lr_back=0.0001, batch_size=128)
    elif args.mode == 'PREPROCESS':
        if args.data_root != '':
            face_crop(args.data_root, device)
            revise_csv(args.data_root)
        if args.extra_data_root != '':
            preprocess_kaggle_dataset(args.extra_data_root, device)