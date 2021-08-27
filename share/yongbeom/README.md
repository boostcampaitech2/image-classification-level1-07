# Yongbeom Kwon T2013's branch

### Model
```bash
model
├── data_loader.py
├── model.py
├── sumit.py
└── train.py
```
실행 방법: 
model dir 내에서 `python train.py` 로 실행하시면 됩니다.

`get_labeled_bootcamp_csv_data.py`: train.csv가 있는 dir 위치를 입력하면 아래의 파일들을 만들어 줍니다.

---
### Data

`train_labeled.csv`: 전체 인원이 포함된 데이터입니다.

`train_labeled.val.csv`: validation data set; 48명에 해당하는 인원이 포함되어 있습니다.

`train_labeled.train.csv`: train data set; validation 인원을 제외한 나머지분들이 포함되어 있습니다.
