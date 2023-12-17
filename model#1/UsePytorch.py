# Pytorch 라이브러리 이용
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Train, Valid 폴더의 상위폴더 경로
PATH = "c:\\sulijoa_ai\\deep-learning-model\\model#1\\dataset"

# GPU 활성화 상태 확인하기
# cuda:0 == GPU, != CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 데이터셋 전처리 (Train / Valid)
data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Train 폴더와 Valid 폴더의 이미지들 전처리
image_datasets = {
    x: datasets.ImageFolder(os.path.join(PATH, x),
                            data_transforms[x]) for x in ['Train', 'Valid']
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=20,
                                   shuffle=True,
                                ) for x in ['Train', 'Valid']
}

# 이미지 개수 확인하기
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Valid']}
print(dataset_sizes)

# 이미지 학습 훈련용 데이터셋 음식 클래스명 가져오기
class_names = image_datasets['Train'].classes
print(class_names)

# 훈련 모델
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 epoch 마다 훈련과 검증 순차적으로 진행
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()  # 모델을 훈련모드로 설정
            else:
                model.eval()   # 모델을 평가모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # 데이터 처리 반복하기?
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                #
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'Valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Valid Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_1 = models.resnet18(pretrained=True) # 나중에 resnet버전 수정을 시도할 수 있을 듯
num_ftrs = model_1.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_1.fc = nn.Linear(num_ftrs, len(class_names))

model_1 = model_1.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_sgd = optim.SGD(model_1.parameters(), lr=0.001, momentum=0.9)
optimizer_adam = optim.Adam(model_1.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_adam, step_size=7, gamma=0.1)

model_resnetft = train_model(model_1, criterion, optimizer_adam, exp_lr_scheduler,
                       num_epochs=15)

# 이미지 예측
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def visualize_single_image(model, image_path, class_names):
    was_training = model.training
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    # Move the tensor to the same device as the model
    device = next(model.parameters()).device  # Get the device of the model
    img_tensor = img_tensor.to(device)  # Move the tensor to that device

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]  # Get the confidence scores

        # Display the image and prediction results
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicted: {class_names[preds[0]]}')

        for i, (class_name, conf) in enumerate(zip(class_names, confidence.cpu())):
            print(f"{class_name}: {conf:.2f}")

        model.train(mode=was_training)

# 학습이 끝난 모델을 저장할 경로 지정
saved_model_path = "c:\\sulijoa_ai\\deep-learning-model\\model#1\\TrainedModel\\saved_model.pth"

# 학습된 모델 저장
torch.save(model_1.state_dict(), saved_model_path)

# Specify the path to the image you want to visualize
image_path = "c:\\sulijoa_ai\\deep-learning-model\\model#1\\dataset\\Valid\\ramen\\5606.jpg"  # Update with the actual path

# Specify the class names
class_names = image_datasets['Train'].classes  # Update with the actual class names

# Visualize the single image
visualize_single_image(model_1, image_path, class_names)