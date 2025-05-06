import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# --------------------------- #
# 1. 설정
# --------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 테스트할 이미지 폴더
test_img_dir = './test_images'  # 여기에 테스트할 이미지 넣기

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------- #
# 2. 모델 불러오기 (VGG16, 5개 클래스)
# --------------------------- #
model = models.vgg16(pretrained=False)

# VGG16은 classifier[-1] 수정
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features, 5)

# 학습된 모델 로드
model.load_state_dict(torch.load('vgg16_best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# --------------------------- #
# 3. 클래스 이름 매핑
# --------------------------- #
class_names = {
    0: 'Albrecht',
    1: 'Edgar',
    2: 'Pablo',
    3: 'Pierre',
    4: 'Vincent'
}

softmax = nn.Softmax(dim=1)

# --------------------------- #
# 4. 테스트 이미지 예측 (확률까지 출력)
# --------------------------- #
test_img_list = os.listdir(test_img_dir)

for img_name in test_img_list:
    img_path = os.path.join(test_img_dir, img_name)
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(image)
        probs = softmax(outputs)
        probs = probs.squeeze()

        predicted_label = torch.argmax(probs).item()
        predicted_class = class_names[predicted_label]
        predicted_prob = probs[predicted_label].item()

    print(f"파일명: {img_name}")
    print(f"➔ 예측 결과: {predicted_class}")
    print(f"➔ 예측 확률: {predicted_prob*100:.2f}%")
    print("-" * 50)
