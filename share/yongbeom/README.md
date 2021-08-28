# Yongbeom Kwon T2013's branch

### Confusion Matrix
```python
from get_confusion_matrix import GetConfusionMatrix

NUM_CLASS = 18
fig_width = 4 + 0.5 * NUM_CLASS
fig_height = 3.5 + 0.44 * NUM_CLASS
## 18개 클래스이면 figsize=(13,12) 추천합니다.

for epoch in range(3):
    age_cm = GetConfusionMatrix(  # @<-----------------
        save_path='cfs_mtx_log',
        current_epoch=epoch,  # 구분점을 epoch으로 두었습니다. (반드시 Epoch일 필요 X)
        n_classes=NUM_CLASS,
        tag='age',  # for multi-model
        # image_name='confusion_matrix',  # default file name
        # only_wrong_label=True,  # wrong label만 표현합니다.
        # count_label=False,  # 수량으로 표현합니다.
        # savefig=False,  # for jupyter-notebook (default: True)
        # showfig=True,  # for jupyter-notebook (default: False)
        figsize=(fig_width, fig_height),  # <- default figsize
        # dpi=200,  # Matplotlib's default is 150 dpi.
        vmax=None)  # A max value of colorbar of heatmap

    for _ in range(3):  # dummy Dataloader
        # train
        target = torch.randint(0, NUM_CLASS, (128, ))
        pred = torch.randint(0, NUM_CLASS, (128, ))

        # prediction
        age_cm.collect_batch_preds(target, pred)  # @<-----------------

    age_cm.epoch_plot()  # @<-----------------
```


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
