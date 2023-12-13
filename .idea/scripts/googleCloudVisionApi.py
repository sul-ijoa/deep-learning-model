from flask import Flask, request, jsonify
import io
import os
import requests
from google.cloud import vision

app = Flask(__name__)

# Google Cloud Vision API 클라이언트 설정
key_file_path = r'C:/lustrous-spirit-407608-d1c17a2817bc.json'
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

        # 이미지에 대해 라벨 감지
        response = client.label_detection(image=image)
        labels = response.label_annotations

        # 감지된 라벨을 내림차순으로 정렬
        sorted_labels = sorted(labels, key=lambda x: x.score, reverse=True)

        # 모든 라벨과 가장 높은 점수의 라벨 출력
        result = {
            '모든 라벨': [label.description for label in labels],
            '가장 높은 점수의 라벨': sorted_labels[0].description if sorted_labels else None
        }

        # HTTP OK(200) HTTP ERROR(500)
        if result['가장 높은 점수의 라벨'] == 'Food':
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)