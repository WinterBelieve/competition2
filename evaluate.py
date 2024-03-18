import torch
from torchvision import transforms
import torch.utils.data
import json
import sys
import os
import zipfile
from torch import nn
from PIL import Image
import glob

# 自定義數據集類，與訓練時使用的一致
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, files, worddict, transform=None):
        self.files = files
        self.worddict = worddict
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.worddict[img_path.split('/')[-1].split('_')[0]]
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model(model, test_loader):
    # 評估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 回傳評估結果
    accuracy = correct / total * 100
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 假設圖像應該被縮放到 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 彩色圖像
])

# 讀取模型路徑並載入模型
model_path = sys.argv[1]
num_classes = 4803  # 指定類別數量
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 64 * 64, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
    nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 解壓縮並準備數據集
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
dataset_dir = os.path.join(base_dir, 'cleaned_data(50_50)')
zip_files = [
    'cleaned_data(50_50)-20200420T071507Z-001.zip',
    'cleaned_data(50_50)-20200420T071507Z-002.zip',
    'cleaned_data(50_50)-20200420T071507Z-003.zip',
    'cleaned_data(50_50)-20200420T071507Z-004.zip'
]
for zip_file in zip_files:
    zip_path = os.path.join(base_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

# 創建詞典來映射標籤到索引
files = glob.glob(os.path.join(dataset_dir, '*.png'))
worddict = {get_label(file): idx for idx, file in enumerate(sorted(set(files)))}

# 創建測試數據集和數據加載器
test_dataset = CustomDataset(files, worddict, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 評估模型
evaluate_model(model, test_loader)