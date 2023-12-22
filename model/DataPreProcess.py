# dataset 전처리 및 train/valid/test 셋으로 분할

import os
import shutil
import random
import json

base_dir = 'model/dataset/category'  # 전처리할 전체 데이터셋 경로
dest_dir = 'model/dataset_preprocessed'  # 전처리 후, train/valid/test 셋으로 분할하여 저장될 폴더 경로

food_index = {} # 음식클래스 라벨링 하기 위한 딕셔너리
splits = ['train', 'valid', 'test']

# train/valid/test 폴더 만들기
os.makedirs(dest_dir, exist_ok=True)

for split in splits:
    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

# train/valid/test 데이터셋 만들고, 음식클래스 라벨링
for category_folder in os.listdir(base_dir): # dataset/category 폴더 가져오기
    category_path = os.path.join(base_dir, category_folder)

    if os.path.isdir(category_path):  # category 폴더가 있는지 확인
        for food_class_folder in os.listdir(category_path): # category 폴더 안에 음식 폴더 가져오기
            food_class_path = os.path.join(category_path, food_class_folder)
            
            if os.path.isdir(food_class_path):  # 음식 폴더가 있는지 확인
                # 각 음식별 라벨링 + 음식이름, 음식카테고리 저장
                food_index[len(food_index)] = {'menu': food_class_folder, 'category': category_folder} # ex) {"0": {"menu": "짬뽕", "category": "chinese"}}

                # 랜덤으로 이미지 추출
                all_images = os.listdir(food_class_path)
                random.shuffle(all_images)

                # 7:2:1 비율로 지정
                train_images = all_images[:int(0.7 * len(all_images))]
                val_images = all_images[int(0.7 * len(all_images)):int(0.9 * len(all_images))] 
                test_images = all_images[int(0.9 * len(all_images)):]

                # 이미지 이름 맨 앞에 음식 인덱스 붙여서 copy하기
                for split, images in zip(['train', 'valid', 'test'], [train_images, val_images, test_images]):
                    os.makedirs(os.path.join(dest_dir, split), exist_ok=True)
                    for img in images:
                        # food_index에서 음식 인덱스 가져오기
                        label = str(len(food_index) - 1)
                        shutil.copy(os.path.join(food_class_path, img), os.path.join(dest_dir, split, f"{label}_{img}"))

# 음식 라벨링 정보 json 파일로 저장하기
# json_path = os.path.join(dest_dir, 'food_label.json')
with open('food_label.json', 'w', encoding='utf-8') as json_file:
    json.dump(food_index, json_file, ensure_ascii=False, indent=4)
