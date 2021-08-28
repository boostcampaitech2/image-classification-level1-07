import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i',
                    '--input-path',
                    metavar='DATA PATH',
                    type=str,
                    help='input data path')
parser.add_argument('-o',
                    '--output',
                    metavar='FILE NAME',
                    type=str,
                    default='train_labeled',
                    help='output file name')
parser.add_argument('-r',
                    '--random-seed',
                    metavar='N',
                    type=int,
                    default=0,
                    help='')
# parser.add_argument('-M',
#                     '--make-from-raw',
#                     metavar='?',
#                     type=bool,
#                     default=False,
#                     help='')
args = parser.parse_args()

# TRAIN_DATA_PATH = '/opt/ml/p_stage_img_cls/data/mask_data/train'
RANDOM_SEED = args.random_seed
TRAIN_DATA_PATH = args.input_path
OUTPUT_FILENAME = args.output
TRAIN_IMAGE_PATH = f"{TRAIN_DATA_PATH}/images"


def fix_issued_label(_df):
    _df.loc[_df['mask_issue'] == True,
            'stem'] = _df.loc[_df['mask_issue'] == True, 'stem'].map({
                'normal':
                'incorrect_mask',
                'incorrect_mask':
                'normal',
            })
    _df.loc[_df['mask_issue'] == True, 'mask_issue'] = False
    _df.loc[_df['gender_issue'] == True,
            'gender'] = _df.loc[_df['gender_issue'] == True, 'gender'].map({
                'female':
                'male',
                'male':
                'female',
            })
    _df.loc[_df['gender_issue'] == True, 'gender_issue'] = False
    return _df


df = pd.read_csv(f"{TRAIN_DATA_PATH}/train.csv")

new_data_list = []
for idx in tqdm(range(len(df))):  # tqdm 을 이용하면 현재 데이터가 얼마나 처리되고 있는지 파악되어 좋습니다.
    _path = df['path'].iloc[idx]  # 순서대로 가져와야 하기 때문에 iloc을 사용해 가져옵니다.
    _gender = df['gender'].iloc[idx]
    _age = df['age'].iloc[idx]
    _id = df['id'].iloc[idx]

    for img_name in Path(f"{TRAIN_IMAGE_PATH}/{_path}").iterdir(
    ):  # 각 dir의 이미지들을 iterative 하게 가져옵니다.
        img_stem = img_name.stem  # 해당 파일의 파일명만을 가져옵니다. 확장자 제외.
        if not img_stem.startswith('._'):  # avoid hidden files
            new_data_list.append(
                [
                    _id, _path, _age, _gender, img_stem,
                    img_name.absolute().__str__()
                ]
            )  # [중요!] id는 중복 문제가 있습니다. https://stages.ai/competitions/74/discussion/post/390

df = pd.DataFrame(new_data_list)
df.columns = ['id', 'path', 'age', 'gender', 'stem', 'img_path']

# https://stages.ai/competitions/74/discussion/post/439 오재환_T2134 캠퍼님
# https://stages.ai/competitions/74/discussion/post/434 서동진_T2108 캠퍼님

gender_labeling_error = [
    '006359', '006360', '006361', '006362', '006363', '006364', '001498-1',
    '004432'
]
mask_labeling_error = ['000020', '004418', '005227']

df['gender_issue'] = df['id'].isin(
    gender_labeling_error)  # gender 의 경우 dir 내의 이미지가 모두 동일 인물이다.
df['mask_issue'] = df['id'].isin(mask_labeling_error) & df['stem'].isin(
    ['normal', 'incorrect_mask'])

wrong_mask_index = df.loc[df['mask_issue'] == True].index

df = fix_issued_label(df)

### GET SCORE
df['label'] = 0

# AGE
# df['label'] += (df['age'] < 30)*0  # Zero sum
df['label'] += ((df['age'] >= 30) & (df['age'] < 60)) * 1
df['label'] += (df['age'] >= 60) * 2

# GENDER
# df['label'] += (df['gender'] == 'male')*0  # Zero sum
df['label'] += (df['gender'] == 'female') * 3

# MASK wearing condition
# df['label'] += (df['stem'].isin(['mask1', 'mask2', 'mask3', 'mask4', 'mask5']))*0  # Zero sum
df['label'] += (df['stem'].isin(['incorrect_mask'])) * 6
df['label'] += (df['stem'].isin(['normal'])) * 12

##
df['mask_label'] = df["stem"].map({
    "normal": 2,
    "incorrect_mask": 1,
    "mask1": 0,
    "mask2": 0,
    "mask3": 0,
    "mask4": 0,
    "mask5": 0,
}).astype(int)

df["gender_label"] = df["gender"].map({"male": 0, "female": 1}).astype(int)

df.loc[df["age"] < 30, "age_label"] = 0
df.loc[(df["age"] >= 30) & (df["age"] < 60), "age_label"] = 1
df.loc[df["age"] >= 60, "age_label"] = 2
df["age_label"] = df["age_label"].astype(int)

df["person_label"] = df["age_label"]
df.loc[df["gender"] == "female", "person_label"] += 3
df["person_label"] = df["person_label"].astype(int)

df.to_csv(f"{TRAIN_DATA_PATH}/train_labeled.csv", index=False)

path_list = []

# LABEL
for i in range(18):
    path_list.extend(
        df.loc[df['label'] == i].drop_duplicates('path')['path'].sample(
            n=8, random_state=RANDOM_SEED).to_list())

## GENDER & AGE
# for i in range(6):
#     path_list.extend(
#         df.loc[df['person_label'] == i].drop_duplicates('path')['path'].sample(
#             n=16, random_state=RANDOM_SEED).to_list())
# n=8).to_list())

path_list = set(path_list)

df.loc[df['path'].isin(path_list)].to_csv(
    f"{TRAIN_DATA_PATH}/{OUTPUT_FILENAME}.r{RANDOM_SEED}.val.csv", index=False)
df.loc[~df['path'].isin(path_list)].to_csv(
    f"{TRAIN_DATA_PATH}/{OUTPUT_FILENAME}.r{RANDOM_SEED}.train.csv",
    index=False)
