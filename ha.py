import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;
import random
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing hypopt library for grid search
from hypopt import GridSearch

# Importing Keras libraries
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.utils import load_img, img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

import warnings
warnings.filterwarnings('ignore')

import os
import shutil

base_dir = '/content/drive/MyDrive/Colab Notebooks/FoodImage'
dest_dir = '/content/drive/MyDrive/Colab Notebooks/food-11'

# Ensure the destination directories exist
os.makedirs(dest_dir, exist_ok=True)

splits = ['training', 'validation', 'test']
for split in splits:
    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

# Define the class mapping
class_map = {'ApplePie': 0, 'BagelSandwich': 1, 'Bibimbop': 2, 'Bread': 3, 'FriedRice': 4, 'Pork': 5}

# Copy the images to the new structure with the desired split
for class_name, class_idx in class_map.items():
    class_folder = os.path.join(base_dir, class_name)
    all_images = os.listdir(class_folder)

    # Shuffle the images to ensure random distribution
    random.shuffle(all_images)

    # Split the images
    train_images = all_images[:int(0.7 * len(all_images))]
    val_images = all_images[int(0.7 * len(all_images)):int(0.9 * len(all_images))]
    test_images = all_images[int(0.9 * len(all_images)):]

    for img in train_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(dest_dir, 'training', f"{class_idx}_{img}"))

    for img in val_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(dest_dir, 'validation', f"{class_idx}_{img}"))

    for img in test_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(dest_dir, 'test', f"{class_idx}_{img}"))

import os

dest_dir = 'c:\\models\\model2\\model#2\\dataset'
splits = ['training', 'validation', 'test']

def extract_classes_from_filenames(filenames):
    # Extract class indices from filenames and return as a set
    return set([int(filename.split('_')[0]) for filename in filenames])

for split in splits:
    split_dir = os.path.join(dest_dir, split)
    all_files = os.listdir(split_dir)
    unique_classes = extract_classes_from_filenames(all_files)

    print(f"In '{split}' directory, unique classes present are: {unique_classes}")