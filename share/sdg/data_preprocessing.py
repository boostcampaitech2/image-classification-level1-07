import os
import pandas as pd
from PIL import Image



def generate_label(train_info ,option):
    idx_num = train_info[train_info['stem']=='.ipynb_checkpoints'].index
    train_info = train_info.drop(idx_num)
    train_image = list(train_info['img_path'])
    if option == 'normal':
        train_label = list(train_info['label'])
        class_num = 18
    elif option == 'age':
        train_label = []
        class_num = 3
        for age_label in train_info['age']:
            if age_label <30:
                train_label.append(0)
            elif 30<= age_label < 60:
                train_label.append(1)
            elif age_label>=60:
                train_label.append(2)
    elif option == 'gender':
        train_label = []
        class_num = 2
        for gender_label in train_info['gender']:
            if gender_label[0]=='m':
                train_label.append(0)
            elif gender_label[0]=='f':
                train_label.append(1)
    elif option == 'mask':
        train_label = []
        class_num = 3
        for mask_label in train_info['stem']:
            if mask_label[0]=='m':
                train_label.append(0)
            elif mask_label[0]=='i':
                train_label.append(1)
            elif mask_label[0]=='n':
                train_label.append(2)  
    return train_image, train_label, class_num

  




