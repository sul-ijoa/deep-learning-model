# model#2 (tensorflow 사용)

# 변수 값 설정 (배치사이즈 8 => 20, epoch 25 => 15 변경)
image = load_img(imagePath, target_size=(224, 224))
batch_size=20
epochs=15

# 결과 (약 20분 소요, 정확도 0.9157)
Epoch 1/15
WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

209/228 [==========================>...] - ETA: 0s - loss: 1.7847 - accuracy: 0.5586   
Epoch 1: val_loss improved from inf to 0.58371, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 2s 3ms/step - loss: 1.7067 - accuracy: 0.5697 - val_loss: 0.5837 - val_accuracy: 0.7885
Epoch 2/15
204/228 [=========================>....] - ETA: 0s - loss: 0.7103 - accuracy: 0.7507
Epoch 2: val_loss improved from 0.58371 to 0.44460, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.6978 - accuracy: 0.7524 - val_loss: 0.4446 - val_accuracy: 0.8342
Epoch 3/15
201/228 [=========================>....] - ETA: 0s - loss: 0.5694 - accuracy: 0.7940
Epoch 3: val_loss improved from 0.44460 to 0.40626, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.5622 - accuracy: 0.7965 - val_loss: 0.4063 - val_accuracy: 0.8475
Epoch 4/15
210/228 [==========================>...] - ETA: 0s - loss: 0.4716 - accuracy: 0.8302
Epoch 4: val_loss improved from 0.40626 to 0.35844, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.4759 - accuracy: 0.8277 - val_loss: 0.3584 - val_accuracy: 0.8620
Epoch 5/15
204/228 [=========================>....] - ETA: 0s - loss: 0.4555 - accuracy: 0.8324
Epoch 5: val_loss improved from 0.35844 to 0.32667, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.4606 - accuracy: 0.8290 - val_loss: 0.3267 - val_accuracy: 0.8804
Epoch 6/15
205/228 [=========================>....] - ETA: 0s - loss: 0.4380 - accuracy: 0.8415
Epoch 6: val_loss did not improve from 0.32667
228/228 [==============================] - 0s 2ms/step - loss: 0.4432 - accuracy: 0.8393 - val_loss: 0.3322 - val_accuracy: 0.8731
Epoch 7/15
209/228 [==========================>...] - ETA: 0s - loss: 0.4034 - accuracy: 0.8507
Epoch 7: val_loss did not improve from 0.32667
228/228 [==============================] - 0s 2ms/step - loss: 0.4105 - accuracy: 0.8472 - val_loss: 0.3449 - val_accuracy: 0.8648
Epoch 8/15
198/228 [=========================>....] - ETA: 0s - loss: 0.3848 - accuracy: 0.8520
Epoch 8: val_loss improved from 0.32667 to 0.29030, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3872 - accuracy: 0.8540 - val_loss: 0.2903 - val_accuracy: 0.8932
Epoch 9/15
210/228 [==========================>...] - ETA: 0s - loss: 0.3669 - accuracy: 0.8583
Epoch 9: val_loss did not improve from 0.29030
228/228 [==============================] - 1s 2ms/step - loss: 0.3744 - accuracy: 0.8549 - val_loss: 0.3168 - val_accuracy: 0.8837
Epoch 10/15
214/228 [===========================>..] - ETA: 0s - loss: 0.3869 - accuracy: 0.8542
Epoch 10: val_loss improved from 0.29030 to 0.27863, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.3872 - accuracy: 0.8544 - val_loss: 0.2786 - val_accuracy: 0.8982
Epoch 11/15
207/228 [==========================>...] - ETA: 0s - loss: 0.3531 - accuracy: 0.8618
Epoch 11: val_loss did not improve from 0.27863
228/228 [==============================] - 1s 2ms/step - loss: 0.3572 - accuracy: 0.8593 - val_loss: 0.2807 - val_accuracy: 0.9021
Epoch 12/15
206/228 [==========================>...] - ETA: 0s - loss: 0.3543 - accuracy: 0.8641
Epoch 12: val_loss improved from 0.27863 to 0.27695, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.3558 - accuracy: 0.8639 - val_loss: 0.2769 - val_accuracy: 0.9015
Epoch 13/15
204/228 [=========================>....] - ETA: 0s - loss: 0.3362 - accuracy: 0.8730
Epoch 13: val_loss improved from 0.27695 to 0.25580, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 3ms/step - loss: 0.3322 - accuracy: 0.8755 - val_loss: 0.2558 - val_accuracy: 0.9137
Epoch 14/15
210/228 [==========================>...] - ETA: 0s - loss: 0.3466 - accuracy: 0.8714
Epoch 14: val_loss improved from 0.25580 to 0.25111, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 3ms/step - loss: 0.3489 - accuracy: 0.8694 - val_loss: 0.2511 - val_accuracy: 0.9132
Epoch 15/15
215/228 [===========================>..] - ETA: 0s - loss: 0.3439 - accuracy: 0.8698
Epoch 15: val_loss improved from 0.25111 to 0.24993, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.3425 - accuracy: 0.8703 - val_loss: 0.2499 - val_accuracy: 0.9182
30/30 [==============================] - 0s 2ms/step

Accuracy on Test Data:  0.9157007376185459

Number of correctly identified imgaes:  869


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.  
