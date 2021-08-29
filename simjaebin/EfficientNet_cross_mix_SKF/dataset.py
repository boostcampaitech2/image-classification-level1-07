from moduleinit import *

def Mix_Up(image1, image2, LAMBDA):
    return (image1.mul(LAMBDA)).add(image2.mul(1-LAMBDA))

def make_dataset_UTK(pre_df, image_path):
    name_list = []
    value_list = []
    for idx, i in enumerate(pre_df['path'].values):
        split_i = i.split('/')[-1]
        name = image_path + '/' + split_i
        name_list.append(name)
    for idx, i in enumerate(pre_df['class'].values):
        value_list.append(int(i / 6))
    pre_df['file'] = name_list
    pre_df['value'] = value_list
    return pre_df


def age_to_classification(age):
    age = int(age)
    if age < 30 :
        trans_age = 0
    elif 30 <= age < 60:
        trans_age = 1
    elif age >= 60:
        trans_age = 2
    return trans_age

def Mix_Up(image1, image2, LAMBDA):
    return (image1.mul(LAMBDA)).add(image2.mul(1-LAMBDA))
#class CustomDataset_train(Dataset):

def make_dataset(pre_df, image_path):
    cnt = [0, 0, 0] 
    MASK = 0
    INCORRECT_MASK = 1
    NORMAL = 2
    path_list = []
    name_list = []
    result = []
    for idx, i in enumerate(pre_df['path'].values):
        for name in os.listdir(image_path + '/' + i):
            if name.find('incorrect') == 0:
                path_list.append(i)
                cnt[INCORRECT_MASK] += 1
                name_list.append(name)
                result.append(INCORRECT_MASK)
            elif name.find('mask') == 0:
                path_list.append(i)
                cnt[MASK] += 1
                name_list.append(name)
                result.append(MASK)
            elif name.find('normal') == 0:
                path_list.append(i)
                cnt[NORMAL] += 1
                name_list.append(name)
                result.append(NORMAL)
            else:
                continue
    train_image_df = pd.DataFrame({'path': path_list, 'file' : name_list, 'value' : result})
    train_result_df = pd.merge(pre_df, train_image_df, how ='outer', on = 'path')
    train_result_df['gender'] = train_result_df['gender'].map({'female' : 1, 'male' : 0})
    train_result_df['age'] = train_result_df['age'].map(age_to_classification)
    train_result_df['class'] = train_result_df['value'] * 6 + train_result_df['gender'] * 3 + train_result_df['age'] 
    return train_result_df



class CustomDataset_UTK(Dataset):
    def __init__(self, transform, data, image_path):
        self.transform = transform
        self.image_path = image_path
        self.pre_df = data
        self.df = make_dataset_UTK(self.pre_df, self.image_path)
        self.df['new_index'] = [i for i in range(len(self.df))]
        self.df.set_index('new_index', inplace = True)
        self.y = self.df['class'].values
        self.column = self.df.columns
        print(self.df.head())

    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):                
        self.labels = self.df['value'][idx]
        image_1_class = self.y[idx]
        image_1_path = self.df['file'][idx]
        image_1 = Image.open(image_1_path)
        y = self.y[idx]
        if self.transform:
            image = self.transform(image_1)
            
        return image, y


class CustomDataset(Dataset):
    def __init__(self, transform, data, image_path):
        self.transform = transform
        self.image_path = image_path
        self.df = data
        self.y = self.df['class'].values
        self.df['new_index'] = [i for i in range(len(self.df))]
        self.df.set_index('new_index', inplace = True)
        self.column = self.df.columns
        print(self.df.head())
        
    def __len__(self):
        return len(self.df)    
    

    def __getitem__(self, idx):
        image_path = '/'.join([train_image, self.df['path'][idx], self.df['file'][idx]])
        image = Image.open(image_path)
        #image = np.array(image)
        X, y = image, self.y[idx]
        if self.transform:
            X = self.transform(X)
        return X, y
        


class MixUpDataset_UTK(Dataset):
    def __init__(self, transform, data, image_path):
        self.transform = transform
        self.image_path = image_path
        self.pre_df = data
        self.df = make_dataset_UTK(self.pre_df, self.image_path)
        self.df['score'] = self.df['class']
        self.df['new_index'] = [i for i in range(len(self.df))]
        self.df.set_index('new_index', inplace = True)
        self.y = self.df['score'].values
        self.column = self.df.columns
        print(self.df.head())
    def __len__(self):
        return len(self.df)    
    

    def __getitem__(self, idx):                
        self.labels = self.df['value'][idx]
        image_1_class = self.y[idx]
        image_1_path = self.df['file'][idx]
        image_1 = Image.open(image_1_path)
        image_2_list = self.df.loc[self.df['value'][idx] != self.df['value']].index
        image_2_idx = random.choice(image_2_list)
        image_2_path = self.df['file'][image_2_idx]
        image_2 = Image.open(image_2_path)
        image_2_class = self.y[image_2_idx]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            
        LAMBDA = np.random.beta(1, 1)
        image_mix = Mix_Up(image_1, image_2, LAMBDA)
        y = torch.Tensor([[image_1_class , LAMBDA], [image_2_class , 1 - LAMBDA]])
        return image_mix, y



class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
