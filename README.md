# image-classification-level1-07
image-classification-level1-07 created by GitHub Classroom
# 2021-08-24
### efficientnet-b1, epoch 10, SGD lr = 0.05,  CE 224, 224
Accuracy = 9~10, F1= 0.05 0.06 언저리

### efficientnet-b1, epoch 10, SGD lr = 0.05,  CE 256, 192
.ipynb, Accuracy = 76.2857, F1 = 0.7141

centercrop 300, 200 보다 resize 256 192가 더 잘나왔음
b0보다 b1이 더 학습 결과가 잘나온다.

# 2021-08-25
tripletmarginloss 잘못짠건지 모르겠는데 loss는 지속적으로 줄어드는 반면, accuracy가 처참함...

### efficientnet-b1, epoch 10, SGD lr = 0.05,  CE
.py,
1. Accuracy = 77.38, F1 = 0.6797
2. Accuracy = 73.7937 F1 = 0.6419

### wideResnet50, epoch 25, Adam lr = 0.001, CE
Accuracy = 75.5238, F1 = 0.6857

### wideResnet50, epoch 10, adam lr = 0.0005, CE, 해상도 512 384
Accuracy = 73.4444, F1 = 0.6255

### wideResnet50, epoch 10, adam lr = 0.0005, CE, 해상도 256 192
Accuracy = 76.2540, F1 = 0.6904
