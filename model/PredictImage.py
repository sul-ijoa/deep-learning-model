import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# 모델 불러오기
model = load_model('1vgg19_finetuned.h5')

test_dir = 'dataset\\test'



# List all image paths in the test directory
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]

# Randomly select an image from the test directory
# random_img_path = random.choice(test_image_paths)
random_img_path = 'dataset\\test\\0_Img_058_0847.jpg'
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0  # Rescale

def get_true_label(img_path):
    return int(os.path.basename(img_path).split("_")[0])

# Load and preprocess the image
img = preprocess_image(random_img_path)

# Make a prediction
predictions = model.predict(img)
predicted_class, confidence = np.argmax(predictions), predictions[0][np.argmax(predictions)]

# Extract the actual label from the filename
actual_label = get_true_label(random_img_path)

# Map class indices to class names
# class_info.json 파일 불러오기
with open('dataset\\class_info.json', 'r', encoding='utf-8') as json_file:
    class_info = json.load(json_file)

# class_info 딕셔너리의 키를 가져와서 리스트에 저장
class_names = list(class_info.keys())

# 리스트 출력
print(class_names)



# Load and preprocess the test images
test_images = [preprocess_image(img_path) for img_path in test_image_paths]

# Convert to a single numpy array
test_images = np.vstack(test_images)

# Extract true labels from the filenames
test_labels = [get_true_label(img_path) for img_path in test_image_paths]
test_labels = np_utils.to_categorical(test_labels, 5)  # Convert to one-hot encoding

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Display the results
plt.imshow(img.squeeze())
plt.title(f"Predicted: {class_names[predicted_class]} (Confidence: {confidence*100:.2f}%)\nActual: {class_names[actual_label]}")
plt.axis('off')
plt.show()
