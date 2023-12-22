from flask import Flask, request, jsonify
import io
import os
import requests
import json
import shutil
import numpy as np
from google.cloud import vision
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from tensorflow.python.keras.utils import np_utils

app = Flask(__name__)

# 음식 라벨 정보 가져오기
food_label_path = 'model/food_label.json'
with open(food_label_path, 'r', encoding='utf-8') as json_file:
    food_label = json.load(json_file)

# 헉습 모델 가져오기
model_path = 'model/1vgg19_finetuned.h5'
model = load_model(model_path)

# 이미지 전처리
def preprocess_image(img_content):
    img = load_img(io.BytesIO(img_content), target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0  # Rescale

# 이미지에서 라벨 가져오기
def get_true_label(img_path):
    return int(os.path.basename(img_path).split("_")[0])

# Google Cloud Vision API 클라이언트 설정
key_file_path = 'C:/GoogleCloudVision-key/service_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
client = vision.ImageAnnotatorClient()

@app.route('/api/image-analyze', methods=['POST'])
def analyze_image():
    try:
        # 요청에서 이미지 파일 가져오기(multipart/form-data)
        image_file = request.files.get('analyzeImg')

        # 이미지를 메모리에 로드
        image_content = image_file.read()
        image = vision.Image(content=image_content)

        # 이미지에 대해 라벨, 오브젝트 감지
        res_label = client.label_detection(image=image)
        res_object = client.object_localization(image=image)
        labels = res_label.label_annotations
        objects = res_object.localized_object_annotations

        # 감지된 라벨을 내림차순으로 정렬
        # sorted_labels = sorted(labels, key=lambda x: x.score, reverse=True)

        # 모든 라벨과 가장 높은 점수의 라벨 출력
        result = {
            '모든 라벨': [label.description for label in labels],
            '모든 오브젝트': [object.name for object in objects],
            # '가장 높은 점수의 라벨': sorted_labels[0].description if sorted_labels else None
        }
        print(result)

        # HTTP OK(200) HTTP ERROR(500)
        if any('Food' in label.description for label in labels) or any('Food'in object.name for object in objects):
            if any('Dessert' in label.description for label in labels) or any('Dessert'in object.name for object in objects):
                return jsonify({'status': 'error', 'message': '디저트로 인식됨'}), 500
            else:
                pred_img = preprocess_image(image_content)
                
                pred_result = model.predict(pred_img)
                pred_index = str(np.argmax(pred_result))
                print("예측 푸드 인덱스 : ", pred_index)
                
                # food_label.json에서 해당 푸드 인덱스에 대한 정보 가져오기
                predicted_result = food_label[pred_index]
                json.dumps(predicted_result, indent=4) if predicted_result else None
                
                return predicted_result, 200
            
        else:
            return jsonify({'status': 'error', 'message': '음식이 아님'}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)