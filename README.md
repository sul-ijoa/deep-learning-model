# model#2 (tensorflow 라이브러리 + 이미지 증강 기법)

# 성능 개선을 위한 변수 설정
learning_rate=0.0001
batch_size=16
batch_size=32
epochs=10

Epoch 1/10
WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\Dayeon\anaconda3\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

110/110 [==============================] - 373s 3s/step - loss: 9.0345 - accuracy: 0.4789 - val_loss: 1.2037 - val_accuracy: 0.8210
Epoch 2/10
110/110 [==============================] - 376s 3s/step - loss: 3.0975 - accuracy: 0.6143 - val_loss: 0.7870 - val_accuracy: 0.8500
Epoch 3/10
110/110 [==============================] - 373s 3s/step - loss: 1.9908 - accuracy: 0.6591 - val_loss: 0.6590 - val_accuracy: 0.8380
Epoch 4/10
110/110 [==============================] - 366s 3s/step - loss: 1.4477 - accuracy: 0.6549 - val_loss: 0.5955 - val_accuracy: 0.8400
Epoch 5/10
110/110 [==============================] - 364s 3s/step - loss: 1.1673 - accuracy: 0.6786 - val_loss: 0.4942 - val_accuracy: 0.8640
Epoch 6/10
110/110 [==============================] - 378s 3s/step - loss: 1.0623 - accuracy: 0.6871 - val_loss: 0.4738 - val_accuracy: 0.8680
Epoch 7/10
110/110 [==============================] - 420s 4s/step - loss: 0.9547 - accuracy: 0.7131 - val_loss: 0.4786 - val_accuracy: 0.8510
Epoch 8/10
110/110 [==============================] - 403s 4s/step - loss: 0.9486 - accuracy: 0.7217 - val_loss: 0.4505 - val_accuracy: 0.8830
Epoch 9/10
110/110 [==============================] - 392s 4s/step - loss: 0.8253 - accuracy: 0.7560 - val_loss: 0.4752 - val_accuracy: 0.8700
Epoch 10/10
110/110 [==============================] - 391s 4s/step - loss: 0.7828 - accuracy: 0.7623 - val_loss: 0.4721 - val_accuracy: 0.8790
16/16 [==============================] - 43s 3s/step - loss: 0.5492 - accuracy: 0.8620
Test Loss: 0.5492034554481506
Test Accuracy: 0.8619999885559082