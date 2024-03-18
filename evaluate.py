import torch
from torch import nn
from torchvision import transforms
import torch.utils.data
import json
import sys
import os
import zipfile
from PIL import Image
import glob

# 自定義數據集類，用於評估模型
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, files, worddict, transform=None):
        self.files = files
        self.worddict = worddict
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')  # 轉換為RGB格式
        label = self.worddict[get_label(img_path)]  # 從文件名提取標籤並轉換為索引
        if self.transform:
            image = self.transform(image)
        return image, label

# 從文件名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

# 評估模型的函數
def evaluate_model(model, test_loader):
    model.eval()  # 將模型設為評估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不計算梯度，節省記憶體和計算資源
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# 圖像轉換操作，與訓練腳本一致
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 假設圖像被縮放到 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 讀取命令列參數中的模型路徑
model_path = sys.argv[1]
# 加載模型，這裡需要與訓練時的模型架構完全一致
num_classes = 4803
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 64 * 64, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
    nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 準備數據集，解壓縮並加載圖像
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
dataset_dir = os.path.join(base_dir, 'cleaned_data(50_50)')
# 解壓.zip文件
for zip_file in ['cleaned_data(50_50)-20200420T071507Z-001.zip', '...']:
    zip_path = os.path.join(base_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

# 創建映射標籤到索引的字典
files = glob.glob(os.path.join(dataset_dir, '*.png'))
worddict = {get_label(file): idx for idx, file in enumerate(sorted(set(get_label(f) for f in files)))}

# 創建數據加載器
test_dataset = CustomDataset(files, worddict, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用數據加載器進行模型評估
evaluate_model(model, test_loader)