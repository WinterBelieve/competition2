import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
import os
import io

# 從檔案名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

# 自定義數據集類
class HandWrite(Dataset):
    def __init__(self, zip_files, worddict, transform=None, start=0, end=0.8):
        self.worddict = worddict
        self.transform = transforms.Compose([transforms.ToTensor()] + (transform if transform else []))
        self.files = []
        self.labels = []

        # 從每個zip文件中讀取圖像
        for zip_filename in zip_files:
            with zipfile.ZipFile(zip_filename, 'r') as z:
                for file in z.namelist():
                    if file.endswith('.png') and get_label(file) in worddict:
                        self.files.append((zip_filename, file))
                        self.labels.append(worddict[get_label(file)])
                        
        # 根據比例分割數據集
        dataset_size = len(self.files)
        self.files = self.files[int(dataset_size*start):int(dataset_size*end)]
        self.labels = self.labels[int(dataset_size*start):int(dataset_size*end)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        zip_filename, filename = self.files[idx]
        label = self.labels[idx]
        # 從zip文件中讀取圖像
        with zipfile.ZipFile(zip_filename, 'r') as z:
            with z.open(filename) as imagefile:
                image = Image.open(imagefile).convert('RGB')
        image = self.transform(image)
        return image, label

# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 獲取 zip 文件的路徑
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
zip_files = [os.path.join(base_dir, f'cleaned_data(50_50)-20200420T071507Z-00{i}.zip') for i in range(1, 5)]

# 創建標籤字典
all_labels = [get_label(filename) for zip_filename in zip_files for filename in zipfile.ZipFile(zip_filename, 'r').namelist() if filename.endswith('.png')]
worddict = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}

# 設定批次大小
BATCH_SIZE = 32

# 創建訓練和測試數據集
train_data = HandWrite(zip_files, worddict, transform=transform, start=0, end=0.8) # 使用前80%的數據作為訓練數據
test_data = HandWrite(zip_files, worddict, transform=transform, start=0.8, end=1.0) # 使用後20%的數據作為測試數據

# 創建 DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# 定義模型架構
num_classes = len(worddict)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 64 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes),
    nn.LogSoftmax(dim=1)
)

# 定義損失函數和優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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