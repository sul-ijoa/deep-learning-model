import os
import numpy as np
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16, VGG19, imagenet_utils
from keras.utils import load_img, img_to_array, to_categorical
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

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
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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

train_x, train_features, train_features_flatten = create_features(train, base_model)
val_x, val_features, val_features_flatten = create_features(val, base_model)
test_x, test_features, test_features_flatten = create_features(test, base_model)

print(train_x.shape, train_features.shape, train_features_flatten.shape)
print(val_x.shape, val_features.shape, val_features_flatten.shape)
print(test_x.shape, test_features.shape, test_features_flatten.shape)


for layer in base_model.layers:
    layer.trainable = False

    model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # Dense에 음식 종류 수
])
    
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(train_x)
history = model.fit(datagen.flow(train_x, y_train, batch_size=8),
                    validation_data=(val_x, y_val),
                    epochs=40,
                    callbacks=[ModelCheckpoint('1vgg19_finetuned.h5', save_best_only=True)])

loss, accuracy = model.evaluate(test_x, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")