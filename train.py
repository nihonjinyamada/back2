import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import os
from torchvision import models
from torchvision.models import ResNet18_Weights

# データの前処理（リサイズ、テンソル化、正規化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データの読み込み
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

# DataLoaderの設定
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet18のロード
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 犬猫の二値分類のため2層に変更
model.fc = nn.Linear(model.fc.in_features, 2)

# GPUを利用できるため設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 二値分類の損失関数
criterion = nn.CrossEntropyLoss()

# 最適化
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 訓練データで訓練
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 勾配をゼロにリセット
        optimizer.zero_grad()

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 訓練の進捗表示
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # 検証
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 訓練後のモデル保存
torch.save(model.state_dict(), 'trained_model/dog_cat_model.pth')


# テストデータで評価
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")