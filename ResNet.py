import pandas as pd
import pymysql
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------- #
# 1. MySQL DB 연결 및 데이터 불러오기 (5진 분류용)
# --------------------------- #
conn = pymysql.connect(
    host='192.168.2.31',
    port=3306,
    user='gimgim',
    password='2457',
    database='flask_project',
    charset='utf8mb4'
)

# 5개 테이블 각각에 label 부여
df_albrecht = pd.read_sql("SELECT filename, artist_id FROM albrecht", conn)
df_albrecht['label'] = 0

df_edgar = pd.read_sql("SELECT filename, artist_id FROM edgar", conn)
df_edgar['label'] = 1

df_pablo = pd.read_sql("SELECT filename, artist_id FROM pablo", conn)
df_pablo['label'] = 2

df_pierre = pd.read_sql("SELECT filename, artist_id FROM pierre", conn)
df_pierre['label'] = 3

df_vincent = pd.read_sql("SELECT filename, artist_id FROM vincent", conn)
df_vincent['label'] = 4

conn.close()

# 전체 데이터 합치기
df_total = pd.concat([df_albrecht, df_edgar, df_pablo, df_pierre, df_vincent], ignore_index=True)
print(df_total.head())

# --------------------------- #
# 2. Dataset 클래스 정의 (변경 없음)
# --------------------------- #
class AlbrechtBinaryDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['filename']
        label = self.dataframe.iloc[idx]['label']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------- #
# 3. 이미지 변환 설정
# --------------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_dir = './All_images'

# --------------------------- #
# 4. 학습용/검증용 데이터 분리
# --------------------------- #
df_train, df_val = train_test_split(df_total, test_size=0.2, stratify=df_total['label'], random_state=42)

print(f"🧩 학습용 데이터 개수: {len(df_train)}장")
print(f"🧪 검증용 데이터 개수: {len(df_val)}장")

train_dataset = AlbrechtBinaryDataset(df_train, img_dir=img_dir, transform=transform)
val_dataset = AlbrechtBinaryDataset(df_val, img_dir=img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --------------------------- #
# 5. 모델 설정 (ResNet18, 5진 분류)
# --------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)

# ResNet18은 fc 수정
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# --------------------------- #
# 6. 학습 + 검증 + EarlyStopping + 시각화용 기록
# --------------------------- #
patience = 3
best_loss = float('inf')
early_stop_counter = 0
num_epochs = 50

# ✅ 시각화용 기록 리스트
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=running_loss/(total//labels.size(0)+1), accuracy=100.*correct/total)

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'resnet18_best_model.pth')
        print(f"✅ Best model saved at Epoch {epoch+1}!")
    else:
        early_stop_counter += 1
        print(f"⏳ No improvement. EarlyStop patience {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("⛔ Early Stopping triggered!")
        break

# --------------------------- #
# 7. 학습 결과 시각화
# --------------------------- #
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig('resnet18_training_curve.png')
plt.show()
