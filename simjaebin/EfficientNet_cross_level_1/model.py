from efficientnet_pytorch import EfficientNet
from moduleinit import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
model = EfficientNet.from_pretrained('efficientnet-b1', num_classes= 18)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), 
                        lr = 0.05,
                        momentum=0.9,
                        weight_decay=1e-4)

lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)