# 모델A(share_jsw)


### Model
```bash
jsw
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
___
Referenced External Datasets:

https://www.kaggle.com/tapakah68/medical-masks-p4

https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset

https://www.kaggle.com/rashikrahmanpritom/age-recognition-dataset

https://www.kaggle.com/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask