import os

import logging

import torch
import torch.optim as optim
# from torch.optim import lr_scheduler

from model import BooDuckMaskModel, device
from data_loader import FaceMaskDataset, get_data_transform

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-8s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M',)
logger = logging.getLogger('train')

# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)

def main():
    MAIN_PATH = '/opt/ml/p_stage_img_cls'
    TRAIN_DATA_PATH = f"{MAIN_PATH}/data/mask_data/train"
    TRAIN_CSV = {
        'train': f"{TRAIN_DATA_PATH}/train_labeled.train.csv",
        'val': f"{TRAIN_DATA_PATH}/train_labeled.val.csv"}

    LEARNING_RATE = 1e-2
    BATCH_SIZE = 24
    NUM_EPOCHS = 30
    TARGET = 'person'
    TIMM_MODEL = 'efficientnetv2_rw_m'

    model_info = {
        'efficientnetv2_rw_t': {'width':224, 'height':224, 'max_size':288},
        'efficientnetv2_rw_s': {'width':288, 'height':288, 'max_size':384},
        'efficientnetv2_rw_m': {'width':320, 'height':320, 'max_size':416}
        }

    IMG_WIDTH = model_info[TIMM_MODEL]['width']
    IMG_HEIGHT = model_info[TIMM_MODEL]['height']
    IMG_MAX_SIZE = model_info[TIMM_MODEL]['max_size']

    logger.info(f"LEARNING_RATE = {LEARNING_RATE}")
    logger.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logger.info(f"NUM_EPOCHS = {NUM_EPOCHS}")
    logger.info(f"TARGET = {TARGET}")
    logger.info(f"TIMM_MODEL = {TIMM_MODEL}")
    logger.info("-----------------------------------")
    logger.info(f"IMG_WIDTH = {IMG_WIDTH}")
    logger.info(f"IMG_HEIGHT = {IMG_HEIGHT}")
    logger.info(f"IMG_MAX_SIZE = {IMG_MAX_SIZE}")
    logger.info("-----------------------------------")

    data_transform = get_data_transform(width=IMG_WIDTH, height=IMG_HEIGHT, max_size=IMG_MAX_SIZE)

    mask_dataset = {x: FaceMaskDataset(csv_file=TRAIN_CSV[x], transform=data_transform[x], target=TARGET) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(mask_dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}

    model = BooDuckMaskModel(model_name=TIMM_MODEL, person_task=True)
    # model.load_state_dict(torch.load('model_weights.pth'))
    model = model.to(device)

    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.train_model(dataloaders=dataloaders, optimizer=opt, num_epochs=NUM_EPOCHS)
    model.save()

if __name__ == '__main__':
    main()
