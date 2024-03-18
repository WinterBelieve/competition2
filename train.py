import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
import zipfile
import os

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

# 從檔案名提取標籤的函數
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

class CustomDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        self.labels = [get_label(file) for file in files]
        self.label_set = list(set(self.labels))
        self.label_to_index = {label: index for index, label in enumerate(self.label_set)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')  # 確保圖像為RGB格式
        label = self.label_to_index[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

# 獲取所有圖像文件路徑
files = glob.glob(f'{dataset_dir}/**/*.png', recursive=True)

# 定義圖像轉換操作
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 創建自定義數據集實例
dataset = CustomDataset(files, transform=transform)

# 分割數據集為訓練集和測試集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 創建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定義模型架構
num_classes = len(dataset.label_set)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 28 * 28, 128),  # 假定圖像為28x28的彩色圖像
    nn.ReLU(),
    nn.Linear(128, num_classes),
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

# 評估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'Test Accuracy: {correct / total * 100}%')

# 保存模型
torch.save(model.state_dict(), 'model.pth')
