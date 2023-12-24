import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;
import tensorflow as tf
import random
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# Importing Keras libraries
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.applications import imagenet_utils, VGG16
from keras.callbacks import ModelCheckpoint
from keras.utils import load_img, img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

import warnings
warnings.filterwarnings('ignore')

base_dir = 'model\\dataset\\category'
dest_dir = 'model\\dataset_preprocessed'

train = [os.path.join("model\\dataset_preprocessed\\train",img) for img in os.listdir("model\\dataset_preprocessed\\train")]
val = [os.path.join("model\\dataset_preprocessed\\valid",img) for img in os.listdir("model\\dataset_preprocessed\\valid")]
test = [os.path.join("model\\dataset_preprocessed\\test",img) for img in os.listdir("model\\dataset_preprocessed\\test")]


train_y = [int(img.split("\\")[-1].split("_")[0]) for img in train]
val_y = [int(img.split("\\")[-1].split("_")[0]) for img in val]
test_y = [int(img.split("\\")[-1].split("_")[0]) for img in test]

# 음식 종류 수
num_class = 5
y_train = np_utils.to_categorical(train_y, num_class)
y_val = np_utils.to_categorical(val_y, num_class)
y_test = np_utils.to_categorical(test_y, num_class)

# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)

def create_features(dataset, pre_model):

    x_scratch = []

    # loop over the images
    for imagePath in dataset:

        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        x_scratch.append(image)

    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=8)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x, features, features_flatten

train_x, train_features, train_features_flatten = create_features(train, model)
val_x, val_features, val_features_flatten = create_features(val, model)
test_x, test_features, test_features_flatten = create_features(test, model)

print(train_x.shape, train_features.shape, train_features_flatten.shape)
print(val_x.shape, val_features.shape, val_features_flatten.shape)
print(test_x.shape, test_features.shape, test_features_flatten.shape)


# model = Sequential()
# model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(6, activation='softmax'))
# model.summary()

# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodellll.best.hdf5',
                               verbose=1,save_best_only=True)
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
# history = model.fit(train_features, y_train, batch_size=8, epochs=15,
#           validation_data=(val_features, y_val), callbacks=[checkpointer],
#           verbose=1, shuffle=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_features, y_train, batch_size=8, epochs=25,
          validation_data=(val_features, y_val), callbacks=[checkpointer],
          verbose=1, shuffle=True)

fig = plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

preds = np.argmax(model.predict(test_features), axis=1)
print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
print("\nNumber of correctly identified imgaes: ",
      accuracy_score(test_y, preds, normalize=False),"\n")
confusion_matrix(test_y, preds, labels=range(0,5))