# 모델B(share_sdg)

### file_path
#### 1. 전처리를 통해 저장된 이미지와 그에 따른 csv파일이 다음과 같은 경로에 있어야 정상적으로 작동이 가능합니다. 

- `combine.csv` : `/opt/ml/input/data/train/combine.csv`

-> 기존 데이터셋(aistages)에 대한 csv 파일과 외부 데이터셋(https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)에 대한 csv 파일을 concat한 csv 파일. csv 파일내에 각 이미지들이 전처리되어 저장된 경로와 class가 들어있고, `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

- `train_labeled_val.csv` : `/opt/ml/input/data/train/train_labeled_val.csv`

-> validation 데이터셋을 위해 학습데이터에서 특정 사람들의 모든 이미지를 빼서 만든 검증 데이터셋에 대한 csv 파일. 양식은 위와 동일하고 마찬가지로 `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

- 기존 데이터셋(aistages)(전처리): `/opt/ml/input/data/train/face_crop_images/`

-> 처음 제공받은 train 파일의 사람 별 디렉토리 형식을 그대로 유지하고 내부 이미지만 전처리된 이미지로 변경(18564장)

- 외부데이터셋(https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)(전처리): `/opt/ml/input/data/train/image_temp/`

-> 해당 디렉토리 안에 전처리된 이미지들이 정상적으로 있으면 됩니다. (31160장)

#### 2. 추론에 사용되는 이미지와 그에 따른 csv파일은 다음과 같은 경로에 있어야 합니다.

- `info.csv` : `/opt/ml/input/data/eval/info.csv`

-> eval 데이터셋의 파일명만 저장되어있는 csv파일. 추론결과를 저장하기 위해 해당 파일명을 읽어서 정답을 추론해 `submission.csv`로 저장하는 바탕이 됩니다.

- eval 데이터셋(aistages)(전처리): `/opt/ml/input/data/eval/face_crop_eval/`

-> 처음 제공받은 eval 이미지 파일들을 전처리된 이미지로 변경 후 저장 (18564장)


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
- 학습시 : sdg dir 내에서 `python main.py` 로 실행하시면 됩니다. 몇 가지 하이퍼파라미터는 `main.py`
안에서 조정이 가능합니다. 학습시 `train.py`를 통해 매 에폭 마다 파라미터와 optimizer state_dict가 저장됩니다. 이는 추후 같은 파라미터로 학습을 재개하기 위한 방법입니다. 

- 추론시 : sdg dir 내에서 `python inference.py` 로 실행하시면 됩니다. 가장 좋은 파라미터 파일을 모델에 불러와 eval 데이터셋에 대한 분류작업을 진행하여 그 결과를 `submission.csv`로 저장합니다. 



