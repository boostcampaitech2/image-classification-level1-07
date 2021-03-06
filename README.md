
<h1 align="center">
<p>image-classification-level1-07
</h1>

<h3 align="center">
<p>권용범_T2013, 서동건_T2107, 심재빈_T2124, 이유진_T2167, 장석우_T2187
</h3>

COVID-19의 감염 확산 방지를 위해서는 공공 장소의 모든 사람이 정확한 마스크 착용을 해야한다. 인적 자원 효율성을 위해 카메라로 비춰진 사람 얼굴 이미지 만으로 올바른 마스크 착용 여부를 판단하는 시스템 개발을 목적으로한다.

## 시연결과

```bash
./infer_mask -i sample/image.png
```
이미지를 입력받아 여러 사람의 얼굴을 인식하여 마스크 착용 유무를 확인합니다.

`-i`: 이미지의 path를 입력합니다.  
`-m1`: 모델1의 model weight 파일을 입력합니다.  
`-m2`: 모델2의 model weight 파일을 입력합니다.  
`-o`: 출력될 directory 위치를 입력합니다. dir는 자동으로 생성됩니다.  

* 원본 이미지:

![](/images/image.png)

* 처리 후:

<img src="/sample/we_are_happy.png" alt="drawing" style="width:700px;"/>

* 추가 이미지:

<p float="left">
  <img src="/sample/image2.crop0.png" width="150" />
  <img src="/sample/image3.crop0.png" width="150" />
</p>

---
## 전처리

아래 모델 A의 실행 방법 : 전처리를 참고

---

## 모델A(model_pseudo_labeling)

<img src="/images/pseudo_labeling.png" alt="drawing" style="background-color:white;"/>


### Model
```bash
model_pseudo_labeling
├── data.py
├── model.py
├── main.py
├── utility.py
└── runners.py
```
파일 설명:

`data.py`: Dataset과 Dataloader가 정의되어 있고 불러오는 함수가 포함 된 파일

`model.py`: 모델이 정의되어 있는 파일

`main.py`: 실행 파일

`utility.py`: Loss 함수, 전처리 함수 등이 포함 된 파일

`runners.py`: train 함수가 포함 된 파일

___
실행 방법: 

`PREPROCESS`: 데이터 전처리
```
python main.py --mode TRAIN --data_root <given data root> --extra_data_root <external dataset root>
```
:bulb:: only input extra data root for `medical mask part 4` dataset

___

`TRAIN`: 모델 학습
```
python main.py --mode TRAIN --data_root <given data root> --train_csv_file <train csv file name> --val_csv_file <validation csv file name> --extra_data_root <external dataset root>
```
`extra_data_root`: 밑의 모든 외부 데이터셋을 모아 놓은 디렉토리

Referenced External Datasets:

https://www.kaggle.com/tapakah68/medical-masks-p4

https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset

https://www.kaggle.com/rashikrahmanpritom/age-recognition-dataset

https://www.kaggle.com/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask

___
## 모델B(model_ViT_Large)

![](/images/vit.png)

### file_path
#### 1. 전처리를 통해 저장된 이미지와 그에 따른 csv파일이 다음과 같은 경로에 있어야 정상적으로 작동이 가능합니다. 

- `combine.csv` : `/opt/ml/input/data/train/combine.csv`

-> 기존 데이터셋(aistages)에 대한 csv 파일과 [외부데이터셋](https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)에 대한 csv 파일을 concat한 csv 파일. csv 파일내에 각 이미지들이 전처리되어 저장된 경로와 class가 들어있고, `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

- `train_labeled_val.csv` : `/opt/ml/input/data/train/train_labeled_val.csv`

-> validation 데이터셋을 위해 학습데이터에서 특정 사람들의 모든 이미지를 빼서 만든 검증 데이터셋에 대한 csv 파일. 양식은 위와 동일하고 마찬가지로 `dataset.py`에서 해당 파일을 판다스 DataFrame 파일로 읽어 데이터를 불러오는 방식입니다.

- 기존 데이터셋(aistages)(전처리): `/opt/ml/input/data/train/face_crop_images/`

-> 처음 제공받은 train 파일의 사람 별 디렉토리 형식을 그대로 유지하고 내부 이미지만 전처리된 이미지로 변경(18564장)

- [외부데이터셋](https://www.kaggle.com/tapakah68/medical-masks-p4?select=df_part_4.csv)(전처리): `/opt/ml/input/data/train/image_temp/`

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

___
### Submission
`ensemble.py`: 
```
python ensemble.py --mode_a <Model A parameter file path> --model_b <Model B parameter file path> --root <evaluation data root directory>
```
실행 방법:

`model_a`: 모델 A 파라미터 파일 경로

`model_b`: 모델 B 파라미터 파일 경로

`root`: evaluation 데이터 경로
