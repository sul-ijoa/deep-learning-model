# model#2 (tensorflow 사용)

# 변수 값 설정 (epoch 15 => 25 변경)
image = load_img(imagePath, target_size=(224, 224))
batch_size=20
epochs=25

# 결과 (약 20분 소요, 정확도 0.9272)
Epoch 1/25
WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

216/228 [===========================>..] - ETA: 0s - loss: 1.6233 - accuracy: 0.5845  
Epoch 1: val_loss improved from inf to 0.78371, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 2s 3ms/step - loss: 1.5808 - accuracy: 0.5917 - val_loss: 0.7837 - val_accuracy: 0.7073
Epoch 2/25
216/228 [===========================>..] - ETA: 0s - loss: 0.7500 - accuracy: 0.7377
Epoch 2: val_loss improved from 0.78371 to 0.52371, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.7481 - accuracy: 0.7370 - val_loss: 0.5237 - val_accuracy: 0.8063
Epoch 3/25
215/228 [===========================>..] - ETA: 0s - loss: 0.5846 - accuracy: 0.7851
Epoch 3: val_loss improved from 0.52371 to 0.42062, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 3ms/step - loss: 0.5808 - accuracy: 0.7857 - val_loss: 0.4206 - val_accuracy: 0.8492
Epoch 4/25
218/228 [===========================>..] - ETA: 0s - loss: 0.5192 - accuracy: 0.8158
Epoch 4: val_loss improved from 0.42062 to 0.38777, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 3ms/step - loss: 0.5198 - accuracy: 0.8149 - val_loss: 0.3878 - val_accuracy: 0.8536
Epoch 5/25
223/228 [============================>.] - ETA: 0s - loss: 0.4651 - accuracy: 0.8278
Epoch 5: val_loss improved from 0.38777 to 0.38355, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 3ms/step - loss: 0.4677 - accuracy: 0.8261 - val_loss: 0.3835 - val_accuracy: 0.8553
Epoch 6/25
226/228 [============================>.] - ETA: 0s - loss: 0.4369 - accuracy: 0.8374
Epoch 6: val_loss improved from 0.38355 to 0.33175, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 1s 2ms/step - loss: 0.4371 - accuracy: 0.8378 - val_loss: 0.3317 - val_accuracy: 0.8731
Epoch 7/25
212/228 [==========================>...] - ETA: 0s - loss: 0.4038 - accuracy: 0.8512
Epoch 7: val_loss improved from 0.33175 to 0.32015, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.4074 - accuracy: 0.8492 - val_loss: 0.3202 - val_accuracy: 0.8737
Epoch 8/25
210/228 [==========================>...] - ETA: 0s - loss: 0.3998 - accuracy: 0.8548
Epoch 8: val_loss improved from 0.32015 to 0.30191, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.4026 - accuracy: 0.8555 - val_loss: 0.3019 - val_accuracy: 0.8826
Epoch 9/25
212/228 [==========================>...] - ETA: 0s - loss: 0.3708 - accuracy: 0.8613
Epoch 9: val_loss improved from 0.30191 to 0.28368, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3716 - accuracy: 0.8619 - val_loss: 0.2837 - val_accuracy: 0.8971
Epoch 10/25
215/228 [===========================>..] - ETA: 0s - loss: 0.3705 - accuracy: 0.8572
Epoch 10: val_loss did not improve from 0.28368
228/228 [==============================] - 0s 2ms/step - loss: 0.3682 - accuracy: 0.8580 - val_loss: 0.3091 - val_accuracy: 0.8804
Epoch 11/25
202/228 [=========================>....] - ETA: 0s - loss: 0.3590 - accuracy: 0.8639
Epoch 11: val_loss improved from 0.28368 to 0.26237, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3638 - accuracy: 0.8628 - val_loss: 0.2624 - val_accuracy: 0.9093
Epoch 12/25
217/228 [===========================>..] - ETA: 0s - loss: 0.3593 - accuracy: 0.8620
Epoch 12: val_loss did not improve from 0.26237
228/228 [==============================] - 0s 2ms/step - loss: 0.3608 - accuracy: 0.8623 - val_loss: 0.2729 - val_accuracy: 0.8993
Epoch 13/25
216/228 [===========================>..] - ETA: 0s - loss: 0.3527 - accuracy: 0.8711
Epoch 13: val_loss did not improve from 0.26237
228/228 [==============================] - 0s 2ms/step - loss: 0.3523 - accuracy: 0.8707 - val_loss: 0.2645 - val_accuracy: 0.9043
Epoch 14/25
217/228 [===========================>..] - ETA: 0s - loss: 0.3375 - accuracy: 0.8747
Epoch 14: val_loss improved from 0.26237 to 0.24951, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3336 - accuracy: 0.8753 - val_loss: 0.2495 - val_accuracy: 0.9082
Epoch 15/25
213/228 [===========================>..] - ETA: 0s - loss: 0.3306 - accuracy: 0.8746
Epoch 15: val_loss did not improve from 0.24951
228/228 [==============================] - 0s 2ms/step - loss: 0.3301 - accuracy: 0.8749 - val_loss: 0.2517 - val_accuracy: 0.9110
Epoch 16/25
218/228 [===========================>..] - ETA: 0s - loss: 0.3406 - accuracy: 0.8732
Epoch 16: val_loss improved from 0.24951 to 0.24830, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3408 - accuracy: 0.8738 - val_loss: 0.2483 - val_accuracy: 0.9071
Epoch 17/25
216/228 [===========================>..] - ETA: 0s - loss: 0.3318 - accuracy: 0.8704
Epoch 17: val_loss improved from 0.24830 to 0.24694, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3345 - accuracy: 0.8711 - val_loss: 0.2469 - val_accuracy: 0.9093
Epoch 18/25
215/228 [===========================>..] - ETA: 0s - loss: 0.3318 - accuracy: 0.8728
Epoch 18: val_loss improved from 0.24694 to 0.24617, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3363 - accuracy: 0.8722 - val_loss: 0.2462 - val_accuracy: 0.9087
Epoch 19/25
216/228 [===========================>..] - ETA: 0s - loss: 0.3155 - accuracy: 0.8775
Epoch 19: val_loss did not improve from 0.24617
228/228 [==============================] - 0s 2ms/step - loss: 0.3159 - accuracy: 0.8771 - val_loss: 0.2497 - val_accuracy: 0.9149
Epoch 20/25
203/228 [=========================>....] - ETA: 0s - loss: 0.3054 - accuracy: 0.8887
Epoch 20: val_loss did not improve from 0.24617
228/228 [==============================] - 0s 2ms/step - loss: 0.3103 - accuracy: 0.8850 - val_loss: 0.2477 - val_accuracy: 0.9110
Epoch 21/25
212/228 [==========================>...] - ETA: 0s - loss: 0.3184 - accuracy: 0.8828
Epoch 21: val_loss improved from 0.24617 to 0.20911, saving model to scratchmodel.best.hdf5
228/228 [==============================] - 0s 2ms/step - loss: 0.3166 - accuracy: 0.8830 - val_loss: 0.2091 - val_accuracy: 0.9304
Epoch 22/25
215/228 [===========================>..] - ETA: 0s - loss: 0.2980 - accuracy: 0.8872
Epoch 22: val_loss did not improve from 0.20911
228/228 [==============================] - 0s 2ms/step - loss: 0.3023 - accuracy: 0.8850 - val_loss: 0.2121 - val_accuracy: 0.9299
Epoch 23/25
215/228 [===========================>..] - ETA: 0s - loss: 0.2826 - accuracy: 0.8963
Epoch 23: val_loss did not improve from 0.20911
228/228 [==============================] - 0s 2ms/step - loss: 0.2868 - accuracy: 0.8948 - val_loss: 0.2228 - val_accuracy: 0.9299
Epoch 24/25
212/228 [==========================>...] - ETA: 0s - loss: 0.2916 - accuracy: 0.8903
Epoch 24: val_loss did not improve from 0.20911
228/228 [==============================] - 0s 2ms/step - loss: 0.3002 - accuracy: 0.8880 - val_loss: 0.2357 - val_accuracy: 0.9132
Epoch 25/25
205/228 [=========================>....] - ETA: 0s - loss: 0.3045 - accuracy: 0.8900
Epoch 25: val_loss did not improve from 0.20911
228/228 [==============================] - 0s 2ms/step - loss: 0.3042 - accuracy: 0.8885 - val_loss: 0.2157 - val_accuracy: 0.9260
30/30 [==============================] - 0s 1ms/step

Accuracy on Test Data:  0.9272918861959958

Number of correctly identified imgaes:  880


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.  
