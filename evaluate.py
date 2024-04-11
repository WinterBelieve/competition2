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

test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# 定義模型架構
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 64 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes),
    nn.LogSoftmax(dim=1)
)

# 加載模型參數
# model_path = '/home/jovyan/competition2/handwrite_model.pth'
model_path = sys.argv[1]  # 模型路徑作為命令行第一個參數傳入
model = torch.jit.load(model_path)

# 模型評估
def evaluate_model(model, test_loader):
    model.eval()  # 設置為評估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在這段代碼中不計算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accurancy": accuracy}))

# 執行模型評估
evaluate_model(model, test_loader)
