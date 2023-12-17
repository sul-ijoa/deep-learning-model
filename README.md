# model#1 (pytorch 라이브러리 사용)

# 변수 값 설정 (배치사이즈 32 => 20 변경)
Train : transforms.RandomResizedCrop(224)
Valid : transforms.Resize(256),
        transforms.CenterCrop(224)
batch_size=32
epoch=15
step_size=7, gamma=0.1

# 결과 (약 60분 소요, 정확도 0.9336)
Epoch 0/14
----------
Train Loss: 1.1246 Acc: 0.5832
Valid Loss: 0.7387 Acc: 0.7280

Epoch 1/14
----------
Train Loss: 0.8336 Acc: 0.6941
Valid Loss: 0.9609 Acc: 0.6544

Epoch 2/14
----------
Train Loss: 0.8071 Acc: 0.6925
Valid Loss: 0.6173 Acc: 0.7696

Epoch 3/14
----------
Train Loss: 0.7262 Acc: 0.7331
Valid Loss: 0.4542 Acc: 0.8408

Epoch 4/14
----------
Train Loss: 0.6900 Acc: 0.7376
Valid Loss: 0.4289 Acc: 0.8448

Epoch 5/14
----------
Train Loss: 0.6640 Acc: 0.7512
Valid Loss: 0.4728 Acc: 0.8216

Epoch 6/14
----------
Train Loss: 0.6403 Acc: 0.7640
Valid Loss: 0.3403 Acc: 0.8816

Epoch 7/14
----------
Train Loss: 0.4789 Acc: 0.8216
Valid Loss: 0.2279 Acc: 0.9256

Epoch 8/14
----------
Train Loss: 0.4410 Acc: 0.8379
Valid Loss: 0.2233 Acc: 0.9224

Epoch 9/14
----------
Train Loss: 0.4162 Acc: 0.8493
Valid Loss: 0.2151 Acc: 0.9216

Epoch 10/14
----------
Train Loss: 0.3811 Acc: 0.8635
Valid Loss: 0.2043 Acc: 0.9240

Epoch 11/14
----------
Train Loss: 0.3774 Acc: 0.8549
Valid Loss: 0.1987 Acc: 0.9320

Epoch 12/14
----------
Train Loss: 0.3646 Acc: 0.8675
Valid Loss: 0.1968 Acc: 0.9288

Epoch 13/14
----------
Train Loss: 0.3555 Acc: 0.8773
Valid Loss: 0.1959 Acc: 0.9296

Epoch 14/14
----------
Train Loss: 0.3499 Acc: 0.8744
Valid Loss: 0.1894 Acc: 0.9336

Training complete in 59m 47s
Best Valid Acc: 0.933600


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.
