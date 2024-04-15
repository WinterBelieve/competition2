import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import json
import sys
import os

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 將圖像大小調整為 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 標準化
])

# 載入測試數據集
data_dir = 'Traditional-Chinese-Handwriting-Dataset/data/cleaned_data(50_50)'
test_data = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(test_data.classes)  # 假設模型和數據集已經準備好
# 假設測試數據集占總數據集的後20%
_, test_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.8), len(test_data)-int(len(test_data)*0.8)])

test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

model_path = sys.argv[1]  # model path
model = torch.jit.load(model_path)

# model evaluation
def evaluate_model(model, test_loader):
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # no gradient
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# evaluate model
evaluate_model(model, test_loader)
