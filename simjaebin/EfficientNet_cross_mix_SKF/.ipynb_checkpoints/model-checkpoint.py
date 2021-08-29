from efficientnet_pytorch import EfficientNet
from moduleinit import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
#model = EfficientNet.from_pretrained('efficientnet-b1', num_classes= 18)
model = torchvision.models.wide_resnet50_2(pretrained=True)
in_feat = model.fc.in_features


model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=in_feat, out_features=18, bias=True),
    torch.nn.Dropout(0.2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), 
                        lr = 0.05,
                        momentum=0.9,
                        weight_decay=1e-4)

lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)