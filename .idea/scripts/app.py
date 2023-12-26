from flask import Flask, request, jsonify
import io
import os
import json
from google.cloud import vision
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

app = Flask(__name__)

# 음식 라벨 정보 가져오기
food_label_path = 'food_label.json'
with open(food_label_path, 'r', encoding='utf-8') as json_file:
    food_label = json.load(json_file)

# num_classes 설정
num_classes = len(food_label)

# ResNet18 모델 정의
model = resnet18(pretrained=False)  # 이 부분을 미리 정의한 모델로 바꿔주세요
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # 이 부분도 데이터셋의 클래스 수에 맞게 수정해주세요

# 학습된 가중치를 로드
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리
def preprocess_image(img_content):
    img = Image.open(io.BytesIO(img_content)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 배치 차원 추가
    return img

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
                
                with torch.no_grad():
                    pred_result = model(pred_img)
                pred_index = torch.argmax(pred_result).item()
                print("예측 푸드 인덱스 : ", pred_index)
                
                # food_label.json에서 해당 푸드 인덱스에 대한 정보 가져오기
                predicted_result = food_label[str(pred_index)]
                json.dumps(predicted_result, indent=4) if predicted_result else None
                
                return jsonify(predicted_result), 200
            
        else:
            return jsonify({'status': 'error', 'message': '음식이 아님'}), 500
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)