import os

import logging

import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

# from torch.optim import lr_scheduler

from model import BooDuckMaskModel, device
from data_loader import FaceMaskDataset, get_data_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-8s %(levelname)-6s %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger("train")


def main():
    MAIN_PATH = "/opt/ml/p_stage_img_cls"
    TRAIN_DATA_PATH = f"{MAIN_PATH}/data/mask_data/train"
    TRAIN_CSV = {
        "train": f"{TRAIN_DATA_PATH}/train_labeled.train.csv",
        "val": f"{TRAIN_DATA_PATH}/train_labeled.val.csv",
    }

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 55
    NUM_EPOCHS = 50
    TARGET = "person"
    TIMM_MODEL = "efficientnetv2_rw_s"

    model_info = {
        "efficientnetv2_rw_t": {
            "width": 224,
            "height": 224,
            "max_size": 288
        },
        "efficientnetv2_rw_s": {
            "width": 288,
            "height": 288,
            "max_size": 384
        },
        "efficientnetv2_rw_m": {
            "width": 320,
            "height": 320,
            "max_size": 416
        },
    }

    IMG_WIDTH = model_info[TIMM_MODEL]["width"]
    IMG_HEIGHT = model_info[TIMM_MODEL]["height"]
    IMG_MAX_SIZE = model_info[TIMM_MODEL]["max_size"]

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

    data_transform = get_data_transform(width=IMG_WIDTH,
                                        height=IMG_HEIGHT,
                                        max_size=IMG_MAX_SIZE)

    mask_dataset = {
        x: FaceMaskDataset(csv_file=TRAIN_CSV[x],
                           transform=data_transform[x],
                           target=TARGET)
        for x in ["train", "val"]
    }
    sampler = {
        "train":
        WeightedRandomSampler(
            [
                mask_dataset["train"].label_weight[_label]
                for _label in mask_dataset["train"].label_list
            ],
            len(mask_dataset["train"]),
        ),
        "val":
        None,
    }
    # dataloaders = {x: torch.utils.data.DataLoader(mask_dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            mask_dataset[x],
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            sampler=sampler[x],
        )
        for x in ["train", "val"]
    }

    model = BooDuckMaskModel(model_name=TIMM_MODEL, person_task=True)
    model.load_state_dict(
        torch.load('backup_effNetV2_CrossEntropy/effNetV2_weights.final.pth'))
    model = model.to(device)

    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.train_model(dataloaders=dataloaders,
                      optimizer=opt,
                      num_epochs=NUM_EPOCHS)
    model.save()


if __name__ == "__main__":
    main()
