import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
import json
import sys
import os

# 從文件名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

# 自定義數據集類
class HandWrite(Dataset):
    def __init__(self, zip_files, worddict, transform=None, start=0, end=1.0):
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.worddict = worddict
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.png') and get_label(name) in self.worddict:
                        self.imgs.append((zip_file, name))
                        self.labels.append(self.worddict[get_label(name)])
        # 計算開始和結束索引
        start_index = int(len(self.imgs) * start)
        end_index = int(len(self.imgs) * end)
        self.imgs = self.imgs[start_index:end_index]
        self.labels = self.labels[start_index:end_index]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        zip_file, img_name = self.imgs[idx]
        label = self.labels[idx]
        with zipfile.ZipFile(zip_file, 'r') as zf:
            with zf.open(img_name) as img_file:
                img = Image.open(img_file).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 獲取 zip 文件的路徑
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
zip_files = [os.path.join(base_dir, f'cleaned_data(50_50)-20200420T071507Z-00{i}.zip') for i in range(1, 5)]
all_labels = [get_label(filename) for zip_filename in zip_files for filename in zipfile.ZipFile(zip_filename, 'r').namelist() if filename.endswith('.png')]
worddict = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}

# 使用後面 20% 的數據集作為測試集
test_data = HandWrite(zip_files, worddict, transform=transform, start=0.8, end=1.0)

# 建立數據加載器
test_loader = DataLoader(test_data, batch_size=32)

# 加載模型
model_path = sys.argv[1]  # 模型路徑作為第一個參數傳入
model = torch.load(model_path)

# 模型評估函數
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# 執行模型評估
evaluate_model(model, test_loader)
