import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import get_model, get_model_list
from data import TestDataset
from trainer import train, k_fold_train
from pytz import timezone
import datetime as dt
from tqdm import tqdm

def load_model(saved_model, num_classes, device):
    model = get_model(args.model_name).to(device)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')

@torch.no_grad()
def inference(test_dir, model_dir, output_dir, args):
    # 모델 저장 및 inference
    # meta 데이터와 이미지 경로를 불러옵니다.
    print('***************Inference Start***************')
    submission = pd.read_csv(os.path.join(args.test_dir, 'info.csv'))
    image_dir = os.path.join(args.test_dir, 'images')
    
    num_classes = 18

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    mean=(0.548, 0.504, 0.479)
    std=(0.237, 0.247, 0.246)
    transform = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    test_dataset = TestDataset(image_paths, transform)

    test_dataloader = DataLoader(dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    
    model = load_model(model_dir, num_classes, device)
    model.eval()
    if not args.k_fold:
        # k_fold 사용하지 않았을 경우 inference
        all_predictions = []
        for images in tqdm(test_dataloader):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
        
    else:
        # k_fold 학습 사용했을 경우 결과 합쳐서 inference
        all_fold_pred = []
        for fold_idx, best_model_dict in enumerate(kfold_models,1):
            model.load_state_dict(best_model_dict)
            model.eval()
            
            each_fold_logits = []
            
            for iter_idx, img in enumerate(test_dataloader, 1):
                with torch.no_grad():
                    img = img.to(device)
                    
                    pred = model(img)
                    each_fold_logits.extend(pred.cpu().tolist())
                    
                    print(f'{fold_idx}/{args.n_fold}folds inference iteration {iter_idx}/{len(test_dataloader)} ', end = '\r')
                    
            all_fold_pred.append(each_fold_logits)
            
        all_fold_pred = np.mean(all_fold_pred, axis = 0)
        all_predictions = np.argmax(all_fold_pred, axis = -1)

    # 제출할 파일을 저장합니다.
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
    submission['ans'] = all_predictions
    csv_name = f'{args.model_name}_{args.epoch_size}_{args.batch_size}_{args.learning_rate}_{args.n_fold}fold_{now}submission.csv'
    submission.to_csv(os.path.join(output_dir, csv_name), index=False) # 모델_epoch_batch_lr_#fold
    print(f'\n test inference csv file : {csv_name} is saved!')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model')) # 모델이 저장된 direc
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/newData'))
    
    parser.add_argument('--k_fold', type = bool, default = False)
    parser.add_argument('--num_workers', type = int, default = 2)
    

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
