# model#2 (tensorflow 사용)

# 변수 값 설정 (기본 값)
image = load_img(imagePath, target_size=(224, 224))
batch_size=8
epochs=25

# 결과 (약 60분 소요, 정확도 0.9251)
Epoch 1/25
WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\envs\ck\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

543/570 [===========================>..] - ETA: 0s - loss: 1.3738 - accuracy: 0.6358  
Epoch 1: val_loss improved from inf to 0.54534, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 2s 3ms/step - loss: 1.3455 - accuracy: 0.6404 - val_loss: 0.5453 - val_accuracy: 0.7858
Epoch 2/25
533/570 [===========================>..] - ETA: 0s - loss: 0.6094 - accuracy: 0.7713
Epoch 2: val_loss improved from 0.54534 to 0.40398, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.6080 - accuracy: 0.7726 - val_loss: 0.4040 - val_accuracy: 0.8392
Epoch 3/25
543/570 [===========================>..] - ETA: 0s - loss: 0.5153 - accuracy: 0.8082
Epoch 3: val_loss improved from 0.40398 to 0.35342, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.5159 - accuracy: 0.8083 - val_loss: 0.3534 - val_accuracy: 0.8625
Epoch 4/25
551/570 [============================>.] - ETA: 0s - loss: 0.4606 - accuracy: 0.8314
Epoch 4: val_loss did not improve from 0.35342
570/570 [==============================] - 1s 1ms/step - loss: 0.4567 - accuracy: 0.8329 - val_loss: 0.4089 - val_accuracy: 0.8503
Epoch 5/25
535/570 [===========================>..] - ETA: 0s - loss: 0.4352 - accuracy: 0.8376
Epoch 5: val_loss improved from 0.35342 to 0.30600, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.4436 - accuracy: 0.8347 - val_loss: 0.3060 - val_accuracy: 0.8965
Epoch 6/25
535/570 [===========================>..] - ETA: 0s - loss: 0.4203 - accuracy: 0.8442
Epoch 6: val_loss improved from 0.30600 to 0.29092, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.4189 - accuracy: 0.8452 - val_loss: 0.2909 - val_accuracy: 0.8987
Epoch 7/25
565/570 [============================>.] - ETA: 0s - loss: 0.3892 - accuracy: 0.8527
Epoch 7: val_loss improved from 0.29092 to 0.28084, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.3898 - accuracy: 0.8523 - val_loss: 0.2808 - val_accuracy: 0.9021
Epoch 8/25
534/570 [===========================>..] - ETA: 0s - loss: 0.3833 - accuracy: 0.8612
Epoch 8: val_loss did not improve from 0.28084
570/570 [==============================] - 1s 1ms/step - loss: 0.3867 - accuracy: 0.8595 - val_loss: 0.3042 - val_accuracy: 0.8887
Epoch 9/25
550/570 [===========================>..] - ETA: 0s - loss: 0.3710 - accuracy: 0.8566
Epoch 9: val_loss improved from 0.28084 to 0.26815, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.3733 - accuracy: 0.8562 - val_loss: 0.2682 - val_accuracy: 0.9060
Epoch 10/25
539/570 [===========================>..] - ETA: 0s - loss: 0.3516 - accuracy: 0.8683
Epoch 10: val_loss did not improve from 0.26815
570/570 [==============================] - 1s 2ms/step - loss: 0.3564 - accuracy: 0.8661 - val_loss: 0.2710 - val_accuracy: 0.8998
Epoch 11/25
547/570 [===========================>..] - ETA: 0s - loss: 0.3488 - accuracy: 0.8688
Epoch 11: val_loss improved from 0.26815 to 0.26282, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.3478 - accuracy: 0.8696 - val_loss: 0.2628 - val_accuracy: 0.9060
Epoch 12/25
546/570 [===========================>..] - ETA: 0s - loss: 0.3618 - accuracy: 0.8645
Epoch 12: val_loss did not improve from 0.26282
570/570 [==============================] - 1s 2ms/step - loss: 0.3664 - accuracy: 0.8632 - val_loss: 0.2755 - val_accuracy: 0.9087
Epoch 13/25
556/570 [============================>.] - ETA: 0s - loss: 0.3399 - accuracy: 0.8698
Epoch 13: val_loss improved from 0.26282 to 0.23711, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.3395 - accuracy: 0.8698 - val_loss: 0.2371 - val_accuracy: 0.9182
Epoch 14/25
557/570 [============================>.] - ETA: 0s - loss: 0.3361 - accuracy: 0.8788
Epoch 14: val_loss did not improve from 0.23711
570/570 [==============================] - 1s 2ms/step - loss: 0.3367 - accuracy: 0.8786 - val_loss: 0.2478 - val_accuracy: 0.9188
Epoch 15/25
519/570 [==========================>...] - ETA: 0s - loss: 0.3366 - accuracy: 0.8752
Epoch 15: val_loss improved from 0.23711 to 0.23024, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 2ms/step - loss: 0.3391 - accuracy: 0.8735 - val_loss: 0.2302 - val_accuracy: 0.9232
Epoch 16/25
541/570 [===========================>..] - ETA: 0s - loss: 0.3235 - accuracy: 0.8771
Epoch 16: val_loss did not improve from 0.23024
570/570 [==============================] - 1s 1ms/step - loss: 0.3290 - accuracy: 0.8751 - val_loss: 0.2414 - val_accuracy: 0.9182
Epoch 17/25
546/570 [===========================>..] - ETA: 0s - loss: 0.3266 - accuracy: 0.8791
Epoch 17: val_loss improved from 0.23024 to 0.23003, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.3291 - accuracy: 0.8775 - val_loss: 0.2300 - val_accuracy: 0.9199
Epoch 18/25
543/570 [===========================>..] - ETA: 0s - loss: 0.3220 - accuracy: 0.8789
Epoch 18: val_loss improved from 0.23003 to 0.22998, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.3239 - accuracy: 0.8786 - val_loss: 0.2300 - val_accuracy: 0.9243
Epoch 19/25
535/570 [===========================>..] - ETA: 0s - loss: 0.3184 - accuracy: 0.8797
Epoch 19: val_loss improved from 0.22998 to 0.22505, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.3192 - accuracy: 0.8786 - val_loss: 0.2250 - val_accuracy: 0.9232
Epoch 20/25
543/570 [===========================>..] - ETA: 0s - loss: 0.3243 - accuracy: 0.8817
Epoch 20: val_loss did not improve from 0.22505
570/570 [==============================] - 1s 1ms/step - loss: 0.3259 - accuracy: 0.8804 - val_loss: 0.2292 - val_accuracy: 0.9210
Epoch 21/25
524/570 [==========================>...] - ETA: 0s - loss: 0.2997 - accuracy: 0.8893
Epoch 21: val_loss improved from 0.22505 to 0.21432, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.3005 - accuracy: 0.8885 - val_loss: 0.2143 - val_accuracy: 0.9304
Epoch 22/25
538/570 [===========================>..] - ETA: 0s - loss: 0.2965 - accuracy: 0.8878
Epoch 22: val_loss did not improve from 0.21432
570/570 [==============================] - 1s 1ms/step - loss: 0.2951 - accuracy: 0.8885 - val_loss: 0.2163 - val_accuracy: 0.9299
Epoch 23/25
533/570 [===========================>..] - ETA: 0s - loss: 0.3027 - accuracy: 0.8870
Epoch 23: val_loss did not improve from 0.21432
570/570 [==============================] - 1s 1ms/step - loss: 0.3044 - accuracy: 0.8878 - val_loss: 0.2349 - val_accuracy: 0.9243
Epoch 24/25
539/570 [===========================>..] - ETA: 0s - loss: 0.2810 - accuracy: 0.8991
Epoch 24: val_loss improved from 0.21432 to 0.19095, saving model to scratchmodel.best.hdf5
570/570 [==============================] - 1s 1ms/step - loss: 0.2800 - accuracy: 0.8997 - val_loss: 0.1909 - val_accuracy: 0.9388
Epoch 25/25
549/570 [===========================>..] - ETA: 0s - loss: 0.2881 - accuracy: 0.8969
Epoch 25: val_loss did not improve from 0.19095
570/570 [==============================] - 1s 1ms/step - loss: 0.2920 - accuracy: 0.8968 - val_loss: 0.2097 - val_accuracy: 0.9338
30/30 [==============================] - 0s 1ms/step

Accuracy on Test Data:  0.9251844046364595

Number of correctly identified imgaes:  878


# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식인지 아닌지 판단
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 분석을 위해 Google Cloud Vision API를 활용합니다.  
가장 높은 확률의 라벨이 "음식"인 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다.  
