# image-classification-level1-07
image-classification-level1-07 created by GitHub Classroom

이미지 디렉토리

훈련 데이터 경로
기존 데이터셋 : /opt/ml/input/data/train/face_crop_images/사람별 폴더 (face crop 된 사진들)
새 데이터셋 : /opt/ml/input/data/train/image_temp

시험 데이터 경로
/opt/ml/input/data/eval/face_crop_eval

combine.csv를 dataset이 읽어서 dataloader를 생성하는 체계
combine.csv 경로 : /opt/ml/input/data/train/combine.csv

inference단계에서는 파일 이름을 처음 주어진 info.csv에서 읽어온 다음 상위 경로와 합쳐서 전체 경로 path만 testdataset에 넣어서 dataloader를 만드는 형태 (main.py 참고)
