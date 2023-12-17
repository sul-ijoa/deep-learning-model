# model#1 (pytorch 라이브러리 사용)

# 변수 값 설정 (직전과 비교하여 이미지 전처리 사이즈 늘림)
Train : transforms.RandomResizedCrop(224)
Valid : transforms.Resize(256),
        transforms.CenterCrop(224)
batch_size=32
epoch=15
step_size=7, gamma=0.1

# 결과 (약 60분 소요, 정확도 0.9392)
Epoch 0/14
----------
Train Loss: 1.0030 Acc: 0.6363
Valid Loss: 0.6442 Acc: 0.7544

Epoch 1/14
----------
Train Loss: 0.7282 Acc: 0.7387
Valid Loss: 0.5866 Acc: 0.7696

Epoch 2/14
----------
Train Loss: 0.6806 Acc: 0.7504
Valid Loss: 0.5766 Acc: 0.7800

Epoch 3/14
----------
Train Loss: 0.6310 Acc: 0.7747
Valid Loss: 0.5581 Acc: 0.8224

Epoch 4/14
----------
Train Loss: 0.6207 Acc: 0.7693
Valid Loss: 0.3827 Acc: 0.8632

Epoch 5/14
----------
Train Loss: 0.5800 Acc: 0.7843
Valid Loss: 0.4334 Acc: 0.8488

Epoch 6/14
----------
Train Loss: 0.5710 Acc: 0.7891
Valid Loss: 0.4196 Acc: 0.8464

Epoch 7/14
----------
Train Loss: 0.3945 Acc: 0.8557
Valid Loss: 0.1971 Acc: 0.9280

Epoch 8/14
----------
Train Loss: 0.3603 Acc: 0.8661
Valid Loss: 0.1935 Acc: 0.9376

Epoch 9/14
----------
Train Loss: 0.3272 Acc: 0.8797
Valid Loss: 0.1860 Acc: 0.9432

Epoch 10/14
----------
Train Loss: 0.3252 Acc: 0.8837
Valid Loss: 0.1683 Acc: 0.9432

Epoch 11/14
----------
Train Loss: 0.3028 Acc: 0.8875
Valid Loss: 0.1688 Acc: 0.9408

Epoch 12/14
----------
Train Loss: 0.3158 Acc: 0.8840
Valid Loss: 0.1781 Acc: 0.9336

Epoch 13/14
----------
Train Loss: 0.3049 Acc: 0.8888
Valid Loss: 0.1804 Acc: 0.9400

Epoch 14/14
----------
Train Loss: 0.2762 Acc: 0.8989
Valid Loss: 0.1730 Acc: 0.9392

Training complete in 60m 38s
Best Valid Acc: 0.943200


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.
