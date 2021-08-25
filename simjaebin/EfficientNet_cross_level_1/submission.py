from torch.utils.data import dataloader
from moduleinit import *
from dataset import CustomDataset, TestDataset
from model import *

if __name__ == "__main__": 
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.



    dataset_test = TestDataset(img_paths = image_paths, 
                                transform = transforms.Compose([
                                    transforms.Resize((256, 192)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))

    loader = DataLoader(dataset_test,
                        batch_size=64,
                        shuffle=False,
                        num_workers=4,
                        )

    print(len(loader))
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('/opt/ml/input/data/train' + 'model.pt'))
    model.eval()
    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.

    all_predictions = []
    for images in loader:
        with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')