# model#2 (tensorflow 사용)

# 변수 값 설정 (배치사이즈 20 => 32, epoch 25 => 15 변경)
image = load_img(imagePath, target_size=(224, 224))
batch_size=32
epochs=15

# 결과 (약 20분 소요, 정확도 0.9220)
Epoch 1/15
WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

139/143 [============================>.] - ETA: 0s - loss: 2.4830 - accuracy: 0.5263   
Epoch 1: val_loss improved from inf to 0.73939, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 3s 9ms/step - loss: 2.4484 - accuracy: 0.5293 - val_loss: 0.7394 - val_accuracy: 0.7195
Epoch 2/15
123/143 [========================>.....] - ETA: 0s - loss: 0.8299 - accuracy: 0.6951
Epoch 2: val_loss improved from 0.73939 to 0.54356, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.8229 - accuracy: 0.6997 - val_loss: 0.5436 - val_accuracy: 0.7896
Epoch 3/15
134/143 [===========================>..] - ETA: 0s - loss: 0.6516 - accuracy: 0.7610
Epoch 3: val_loss improved from 0.54356 to 0.46899, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.6529 - accuracy: 0.7592 - val_loss: 0.4690 - val_accuracy: 0.8236
Epoch 4/15
142/143 [============================>.] - ETA: 0s - loss: 0.5569 - accuracy: 0.7960
Epoch 4: val_loss improved from 0.46899 to 0.39908, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.5566 - accuracy: 0.7960 - val_loss: 0.3991 - val_accuracy: 0.8592
Epoch 5/15
133/143 [==========================>...] - ETA: 0s - loss: 0.5056 - accuracy: 0.8170
Epoch 5: val_loss improved from 0.39908 to 0.36308, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.5080 - accuracy: 0.8138 - val_loss: 0.3631 - val_accuracy: 0.8731
Epoch 6/15
130/143 [==========================>...] - ETA: 0s - loss: 0.4642 - accuracy: 0.8293
Epoch 6: val_loss improved from 0.36308 to 0.35305, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.4639 - accuracy: 0.8292 - val_loss: 0.3530 - val_accuracy: 0.8748
Epoch 7/15
140/143 [============================>.] - ETA: 0s - loss: 0.4394 - accuracy: 0.8384
Epoch 7: val_loss improved from 0.35305 to 0.33200, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.4408 - accuracy: 0.8371 - val_loss: 0.3320 - val_accuracy: 0.8792
Epoch 8/15
135/143 [===========================>..] - ETA: 0s - loss: 0.4272 - accuracy: 0.8387
Epoch 8: val_loss improved from 0.33200 to 0.30772, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.4260 - accuracy: 0.8400 - val_loss: 0.3077 - val_accuracy: 0.8887
Epoch 9/15
135/143 [===========================>..] - ETA: 0s - loss: 0.3934 - accuracy: 0.8586
Epoch 9: val_loss improved from 0.30772 to 0.29975, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 6ms/step - loss: 0.3943 - accuracy: 0.8569 - val_loss: 0.2998 - val_accuracy: 0.8870
Epoch 10/15
131/143 [==========================>...] - ETA: 0s - loss: 0.3876 - accuracy: 0.8564
Epoch 10: val_loss improved from 0.29975 to 0.29024, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.3878 - accuracy: 0.8573 - val_loss: 0.2902 - val_accuracy: 0.8987
Epoch 11/15
138/143 [===========================>..] - ETA: 0s - loss: 0.3704 - accuracy: 0.8644
Epoch 11: val_loss did not improve from 0.29024
143/143 [==============================] - 1s 5ms/step - loss: 0.3769 - accuracy: 0.8628 - val_loss: 0.3099 - val_accuracy: 0.8909
Epoch 12/15
135/143 [===========================>..] - ETA: 0s - loss: 0.3718 - accuracy: 0.8630
Epoch 12: val_loss did not improve from 0.29024
143/143 [==============================] - 1s 5ms/step - loss: 0.3706 - accuracy: 0.8634 - val_loss: 0.2943 - val_accuracy: 0.8965
Epoch 13/15
136/143 [===========================>..] - ETA: 0s - loss: 0.3620 - accuracy: 0.8716
Epoch 13: val_loss improved from 0.29024 to 0.27649, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 6ms/step - loss: 0.3623 - accuracy: 0.8714 - val_loss: 0.2765 - val_accuracy: 0.9015
Epoch 14/15
132/143 [==========================>...] - ETA: 0s - loss: 0.3606 - accuracy: 0.8662
Epoch 14: val_loss improved from 0.27649 to 0.27470, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 5ms/step - loss: 0.3594 - accuracy: 0.8665 - val_loss: 0.2747 - val_accuracy: 0.9037
Epoch 15/15
133/143 [==========================>...] - ETA: 0s - loss: 0.3486 - accuracy: 0.8745
Epoch 15: val_loss improved from 0.27470 to 0.25564, saving model to scratchmodel.best.hdf5
143/143 [==============================] - 1s 6ms/step - loss: 0.3562 - accuracy: 0.8735 - val_loss: 0.2556 - val_accuracy: 0.9076
30/30 [==============================] - 0s 4ms/step

Accuracy on Test Data:  0.9220231822971549

Number of correctly identified imgaes:  875


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.  
