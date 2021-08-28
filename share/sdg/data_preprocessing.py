import os
import pandas as pd
from PIL import Image



def generate_label(train_info ,option):
    train_image = list(train_info['img_path'])
    if option == 'normal':
        train_label = list(train_info['class'])
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
            if 'incorrect_mask' in mask_label:
                train_label.append(1)
            elif 'normal' in mask_label:
                train_label.append(2)
            else:
                train_label.append(0) 
    return train_image, train_label, class_num

  




