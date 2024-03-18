import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
import zipfile
import os

# 從檔案名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

# 自定義數據集類
class HandWrite(Dataset):
    def __init__(self, files, worddict, transform=None):
        self.files = files
        self.worddict = worddict
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')  # 確保為RGB格式
        label = self.worddict[get_label(img_path)]  # 從檔案名提取標籤並轉換為索引
        if self.transform:
            image = self.transform(image)
        return image, label

# 解壓數據集
zip_files = [
    'cleaned_data(50_50)-20200420T071507Z-001.zip',
    'cleaned_data(50_50)-20200420T071507Z-002.zip',
    'cleaned_data(50_50)-20200420T071507Z-003.zip',
    'cleaned_data(50_50)-20200420T071507Z-004.zip'
]
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
dataset_dir = os.path.join(base_dir, 'cleaned_data(50_50)')

# 確保資料集目錄存在
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# 解壓所有.zip文件
for zip_file in zip_files:
    zip_path = os.path.join(base_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

# 獲取所有圖像文件路徑
files = glob.glob(f'{dataset_dir}/**/*.png', recursive=True)

# 創建標籤字典
worddict = {label: idx for idx, label in enumerate(sorted(set([get_label(f) for f in files])))}

# 定義轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 將圖片縮放為 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 根據訓練集和測試集的切割創建自定義數據集實例
train_data = HandWrite(files, worddict, transform)
test_data = HandWrite(files, worddict, transform)

# 分割訓練集和測試集
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

# 創建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定義模型架構
num_classes = len(worddict)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 64 * 3, 1024),  # 假定圖像為64x64的彩色圖像
    nn.ReLU(),
    nn.Linear(1024, num_classes),
    nn.LogSoftmax(dim=1)
)

# 定義損失函數和優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Loss: {total_loss / total}, Accuracy: {accuracy}%')

# 保存模型
torch.save(model.state_dict(), 'handwrite_model.pth')