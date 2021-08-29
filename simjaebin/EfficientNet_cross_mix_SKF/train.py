from torch.utils.data import dataloader
from moduleinit import *
from dataset import CustomDataset,MixUpDataset_UTK, CustomDataset_UTK, TestDataset
from dataset import make_dataset
from model import *
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import wandb


def train(model, criterion, optimizer, scheduler, dataloader, num_epochs=25, Trained = False):
    config = {"epochs" : num_epochs, "batch_size" : BATCH_SIZE, "learning_rate" : LEARNING_RATE}
    wandb.init(project = '123', config = config)
    if Trained == True:
        model.load_state_dict(torch.load('/opt/ml/input/data/train' + model_name))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            F1_score = 0
            # Iterate over data.
            cnt = 0
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim = -1)
                    loss = criterion(outputs, labels)
                    if cnt == 0:
                        print(loss)

                    cnt += 1
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                F1_score += f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='macro')  
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            epoch_f1 = F1_score / dataloader[phase].__len__()
            if phase == 'valid':
                wandb.log({'accuracy' : epoch_loss, 'loss': epoch_acc, 'F1-score': epoch_f1})

            print('{} Loss: {:.2f} Acc: {:.1f} F1 : {:.2f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))
           
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))
    print('Best valid F1: %d - %.1f' %(best_idx, best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), '/opt/ml/input/data/train' + model_name)
    print('model saved')
    return model

def mixup_criterion(criterion, output, label):
    y1 = torch.as_tensor(label[:, 0, 0].clone().detach().requires_grad_(True), dtype= torch.int64)
    y2 = torch.as_tensor(label[:, 1, 0].clone().detach().requires_grad_(True), dtype= torch.int64)
    return 0.2 * criterion(output, y1) + 0.8 * criterion(output, y2)


def train_mix(model, criterion, optimizer, scheduler, dataloader, num_epochs=25, Trained = False):
    #config = {"epochs" : num_epochs, "batch_size" : BATCH_SIZE, "learning_rate" : LEARNING_RATE}
    #wandb.init(project = '123', config = config)
    if Trained == True:
        model.load_state_dict(torch.load('/opt/ml/input/data/train' + model_name))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            F1_score = 0
            # Iterate over data.
            cnt = 0
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    loss = mixup_criterion(criterion, outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
                #F1_score += f1_score(preds.cpu().detach().numpy(), outputs.cpu().detach().numpy(), average='macro')  
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            #epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            epoch_acc = 0
            #epoch_f1 = F1_score / dataloader.__len__()
            epoch_f1 = 0
            #if phase == 'valid':
                #wandb.log({'accuracy' : epoch_loss, 'loss': epoch_acc, 'F1-score': epoch_f1})

            print('{} Loss: {:.2f} Acc: {:.1f} F1 : {:.2f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))
           
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    #print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))
    #print('Best valid F1: %d - %.1f' %(best_idx, best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), '/opt/ml/input/data/train' + model_name)
    print('model saved')
    return model

def classification_class(i):
    return int(i/100)

def train_image_dataset(K, set_epoch, Trained = False):
    df = pd.read_csv(path_df)
    df = make_dataset(df, train_image)
    class_list = []
    class_distribution = []
    class_num = [0 for i in range(18)]
    for idx, value in enumerate(df['class']):
        class_list.append(idx)
        class_distribution.append(value)
        class_num[value] += 1

    min_class = min(class_num)
    K = int(min_class / 8)
    SKF = StratifiedKFold(n_splits=K, shuffle= True)
    SKF.split(class_list, class_distribution)

    for train_index, valid_index in SKF.split(class_list, class_distribution):

        datasets = {}
        datasets['train'] = CustomDataset(transform = transforms.Compose([
                                    transforms.Resize(Imagesize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df.iloc[train_index], image_path = train_image)

        datasets['valid'] = CustomDataset(transform = transforms.Compose([
                                    transforms.Resize(Imagesize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df.iloc[valid_index], image_path = train_image)

        print(len(datasets['train']), len(datasets['valid']))
        #dataloader
        dataloaders= {}

        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )

        dataloaders['valid'] = DataLoader(dataset=datasets['valid'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )
        #train model
        _ = train(model, criterion, optimizer, scheduler, dataloaders, num_epochs=set_epoch, Trained= Trained)
        #fold 1
        break


def train_UTK_dataset_mix(K, set_epoch, Trained = False):
    df_UTK = pd.read_csv(path_df_UTK)
    class_list = []
    class_distribution = []
    class_num = [0 for i in range(18)]
    for idx, value in enumerate(df_UTK['class']):
        class_list.append(idx)
        class_distribution.append(value)
        class_num[value] += 1

    min_class = min(class_num)
    K = int(min_class / 8)
    SKF = StratifiedKFold(n_splits=K, shuffle= True)
    SKF.split(class_list, class_distribution)

    for train_index, valid_index in SKF.split(class_list, class_distribution):
        datasets = {}
        datasets['train'] = MixUpDataset_UTK(transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df_UTK.iloc[train_index], image_path = train_image_UTK)

        datasets['valid'] = MixUpDataset_UTK(transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df_UTK.iloc[valid_index], image_path = train_image_UTK)
        print(len(datasets['train']), len(datasets['valid']))
        #dataloader
        dataloaders= {}

        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )

        dataloaders['valid'] = DataLoader(dataset=datasets['valid'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )

        _ = train_mix(model, criterion, optimizer, scheduler, dataloaders, num_epochs=set_epoch, Trained = Trained)
        #fold 1
        break

def train_UTK_dataset(K, set_epoch, Trained = False):
    df_UTK = pd.read_csv(path_df_UTK)
    class_list = []
    class_distribution = []
    class_num = [0 for i in range(18)]
    for idx, value in enumerate(df_UTK['class']):
        class_list.append(idx)
        class_distribution.append(value)
        class_num[value] += 1

    min_class = min(class_num)
    K = int(min_class / 8)
    SKF = StratifiedKFold(n_splits=K, shuffle= True)
    SKF.split(class_list, class_distribution)
    for train_index, valid_index in SKF.split(class_list, class_distribution):
        datasets = {}
        datasets['train'] = CustomDataset_UTK(transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df_UTK.iloc[train_index], image_path = train_image_UTK)

        datasets['valid'] = CustomDataset_UTK(transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]), data = df_UTK.iloc[valid_index], image_path = train_image_UTK)
        print(len(datasets['train']), len(datasets['valid']))
        #dataloader
        dataloaders= {}

        dataloaders['train'] = DataLoader(dataset=datasets['train'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )

        dataloaders['valid'] = DataLoader(dataset=datasets['valid'],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=4,
                                        )

        _ = train(model, criterion, optimizer, scheduler, dataloaders, num_epochs=set_epoch, Trained = Trained)
        #fold 1
        break