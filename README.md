# model#1 (pytorch 라이브러리 사용)

# 변수 값 설정
Train : transforms.RandomResizedCrop(128)
Valid : transforms.Resize(128),
        transforms.CenterCrop(128)
batch_size=32
epoch=15
step_size=7, gamma=0.1

# 결과 (약 34분 소요, 정확도 0.9120)
Epoch 0/14
----------
Train Loss: 1.0536 Acc: 0.6093
Valid Loss: 0.9376 Acc: 0.6768

Epoch 1/14
----------
Train Loss: 0.8165 Acc: 0.6880
Valid Loss: 0.7091 Acc: 0.7568

Epoch 2/14
----------
Train Loss: 0.7264 Acc: 0.7309
Valid Loss: 0.5631 Acc: 0.7768

Epoch 3/14
----------
Train Loss: 0.6656 Acc: 0.7499
Valid Loss: 0.5027 Acc: 0.8192

Epoch 4/14
----------
Train Loss: 0.6187 Acc: 0.7771
Valid Loss: 0.5677 Acc: 0.7952

Epoch 5/14
----------
Train Loss: 0.5860 Acc: 0.7787
Valid Loss: 0.7624 Acc: 0.7856

Epoch 6/14
----------
Train Loss: 0.5889 Acc: 0.7909
Valid Loss: 1.0395 Acc: 0.6952

Epoch 7/14
----------
Train Loss: 0.4719 Acc: 0.8293
Valid Loss: 0.2968 Acc: 0.8904

Epoch 8/14
----------
Train Loss: 0.4047 Acc: 0.8483
Valid Loss: 0.2679 Acc: 0.9064

Epoch 9/14
----------
Train Loss: 0.3739 Acc: 0.8651
Valid Loss: 0.2831 Acc: 0.8976

Epoch 10/14
----------
Train Loss: 0.3651 Acc: 0.8680
Valid Loss: 0.2745 Acc: 0.9000

Epoch 11/14
----------
Train Loss: 0.3603 Acc: 0.8699
Valid Loss: 0.2678 Acc: 0.9080

Epoch 12/14
----------
Train Loss: 0.3354 Acc: 0.8771
Valid Loss: 0.2525 Acc: 0.9136

Epoch 13/14
----------
Train Loss: 0.3153 Acc: 0.8901
Valid Loss: 0.2605 Acc: 0.9120

Epoch 14/14
----------
Train Loss: 0.3056 Acc: 0.8899
Valid Loss: 0.2529 Acc: 0.9120

Training complete in 33m 54s
Best Valid Acc: 0.913600


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.
