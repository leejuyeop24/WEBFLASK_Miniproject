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
# 2. 모델 불러오기 (5개 클래스용)
# --------------------------- #
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # ✅ 5진 분류

model.load_state_dict(torch.load('best_model.pth', map_location=device))
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
# 4. 테스트 이미지 예측 (확률까지)
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
