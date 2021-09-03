
from final_code.data import get_data_loader
from utility import *



def train_runner(train_dataset,
                 val_dataset,
                 device,
                 model,
                 criterion,
                 batch_size=64,
                 EPOCHS=40,
                 lr1=0.05,
                 lr2=0.0002,
                 ViT=False,
                 num_classes=18,
                 cutmix=False,
                 BETA=1.0):
    sampler = get_weighted_random_sampler(num_classes, train_dataset, cutmix)
    train_loader = get_data_loader(train_dataset, sampler, batch_size)
    
    optimizer = get_optimizer(model, lr1, lr2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    model = model.to(device)
