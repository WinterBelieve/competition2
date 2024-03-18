import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
import zipfile
import os

# 定義從檔案名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

class HandWrite(Dataset):
    def __init__(self, files, worddict, transform=None, start=0, end=0.8):
        self.worddict = worddict
        self.files = files[int(len(files) * start):int(len(files) * end)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.worddict[get_label(img_path)]
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

for zip_file in zip_files:
    zip_path = os.path.join(base_dir, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

# 獲取所有圖像文件路徑
files = glob.glob(f'{dataset_dir}/*.png')

# 創建標籤字典
worddict = {get_label(file): idx for idx, file in enumerate(sorted(set(files)))}

# 定義圖像轉換操作
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 創建數據集實例
train_data = HandWrite(files, worddict, transform, end=0.8)
test_data = HandWrite(files, worddict, transform, start=0.8)

# 創建 DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

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
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f'Epoch {epoch+1}, Loss: {total_loss / total}, Accuracy: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), 'handwrite_model.pth')
