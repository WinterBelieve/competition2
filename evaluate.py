import torch
from torch import nn
from torchvision import transforms
import torch.utils.data
import json
import sys
from PIL import Image
import zipfile
import io

# 從檔案名提取標籤的函數
def get_label(filename):
    return filename.split('_')[0]

# 自定義數據集類，用於直接從zip文件讀取數據
class HandWrite(Dataset):
    def __init__(self, zip_files, worddict, transform=None):
        self.zip_files = zip_files
        self.worddict = worddict
        self.transform = transform
        self.images = []
        for zip_filename in zip_files:
            with zipfile.ZipFile(zip_filename, 'r') as z:
                for file in z.namelist():
                    if file.endswith('.png'):
                        self.images.append((zip_filename, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        zip_filename, filename = self.images[idx]
        with zipfile.ZipFile(zip_filename, 'r') as z:
            with z.open(filename) as imagefile:
                image = Image.open(imagefile).convert('RGB')
        label_name = get_label(filename)
        label = self.worddict[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 讀取模型路徑並載入模型
model_path = sys.argv[1]
model = torch.load(model_path, map_location=torch.device('cpu'))

# 獲取 zip 文件的路徑
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
zip_files = [os.path.join(base_dir, f'cleaned_data(50_50)-20200420T071507Z-00{i}.zip') for i in range(1, 5)]

# 創建標籤字典
all_labels = [get_label(filename) for zip_filename in zip_files for filename in zipfile.ZipFile(zip_filename, 'r').namelist() if filename.endswith('.png')]
worddict = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}

# 創建測試數據集和數據加載器
test_dataset = HandWrite(zip_files, worddict, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 使用數據加載器進行模型評估
evaluate_model(model, test_loader)
