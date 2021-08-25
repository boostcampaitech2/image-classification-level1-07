from moduleinit import *


def age_to_classification(age):
    age = int(age)
    if age < 30 :
        trans_age = 0
    elif 30 <= age < 60:
        trans_age = 1
    elif age >= 60:
        trans_age = 2
    return trans_age

class CustomDataset(Dataset):
    def __init__(self, transform, path, image_path):
        self.path = path
        self.transform = transform
        self.image_path = image_path
        self.df = self.make_dataset()
        self.df['score'] = self.df['value'] * 6 + self.df['gender'] * 3 + self.df['age'] 
        self.y = self.df['score'].values
        self.column = self.df.columns

        
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
        

    def make_dataset(self):
        cnt = [0, 0, 0] 
        MASK = 0
        INCORRECT_MASK = 1
        NORMAL = 2
        path_list = []
        name_list = []
        result = []
        train_df = pd.read_csv(self.path)
        for idx, i in enumerate(train_df['path'].values):
            for name in os.listdir(self.image_path + '/' + i):
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
        train_result_df = pd.merge(train_df, train_image_df, how ='outer', on = 'path')
        train_result_df['gender'] = train_result_df['gender'].map({'female' : 1, 'male' : 0})
        train_result_df['age'] = train_result_df['age'].map(age_to_classification)
        return train_result_df

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