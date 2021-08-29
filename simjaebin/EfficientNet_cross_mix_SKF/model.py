from efficientnet_pytorch import EfficientNet
from moduleinit import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
model = EfficientNet.from_pretrained('efficientnet-b1', num_classes= 18)

'''model = torchvision.models.wide_resnet50_2(pretrained=True)
in_feat = model.fc.in_features


model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=in_feat, out_features=18, bias=True),
    torch.nn.Dropout(0.2)
)'''
LEARNING_RATE =  0.05
model = model.to(device)
BATCH_SIZE = 64
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                        lr = LEARNING_RATE,
                        momentum=0.9,
                        weight_decay=1e-4)

lmbda = lambda epoch: 0.98739
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
