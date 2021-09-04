# 모델B(share_sdg)

### file_path
전처리를 통해 저장된 이미지와 csv파일이 다음과 같은 경로에 있어야 정상적으로 작동이 가능합니다. 

`combine.csv` : 

-> 기존 데이터셋(aistages)에 대한 csv 파일과 외부 데이터셋(https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)에 대한 csv 파일을 concat한 csv 파일. csv 파일내에 각 이미지들이 전처리되어 저장된 경로와 class가 들어있고, `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

`train_labeled_val.csv` : validation 데이터셋을 위해 학습데이터에서 특정 사람들의 모든 이미지를 빼서 만든 검증 데이터셋에 대한 csv 파일. 양식은 위와 동일하고 마찬가지로 `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

기존 데이터셋(aistages)(전처리): `/opt/ml/input/data/train/face_crop_images/`

-> 처음 제공받은 train 파일의 사람 별 디렉토리 형식을 그대로 유지하고 내부 이미지만 전처리된 이미지로 변경(18564장)

외부데이터셋(https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)(전처리): `/opt/ml/input/data/train/image_temp/`

-> 해당 디렉토리 안에 전처리된 이미지들이 정상적으로 있으면 됩니다. (31160장)

### Model
```bash
sdg
├── dataset.py
├── model_normal.py
├── train.py
├── main.py
└── inference.py

```
실행 방법: 
model dir 내에서 `python train.py` 로 실행하시면 됩니다.

`get_labeled_bootcamp_csv_data.py`: train.csv가 있는 dir 위치를 입력하면 아래의 파일들을 만들어 줍니다.

---
### Data

`train_labeled.csv`: 전체 인원이 포함된 데이터입니다.

`train_labeled.val.csv`: validation data set; 48명에 해당하는 인원이 포함되어 있습니다.

`train_labeled.train.csv`: train data set; validation 인원을 제외한 나머지분들이 포함되어 있습니다.
