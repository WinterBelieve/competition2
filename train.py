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
    return filename.split('_')[0]

# 自定義數據集類
class HandWrite(Dataset):
    def __init__(self, zip_files, worddict, transform=None, start_ratio=0, end_ratio=0.8):
        self.worddict = worddict
        self.transform = transform
        self.image_info = []
        # 從每個ZIP文件中讀取PNG文件
        for zip_filename in zip_files:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                # 這裡假設ZIP文件中的每個文件都是圖片文件
                name_list = zip_ref.namelist()
                # 根據比例分割數據
                start_idx = int(len(name_list) * start_ratio)
                end_idx = int(len(name_list) * end_ratio)
                for filename in name_list[start_idx:end_idx]:
                    if filename.endswith('.png'):
                        self.image_info.append((zip_ref, filename))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        zip_ref, filename = self.image_info[idx]
        label_name = get_label(os.path.basename(filename))
        label = self.worddict[label_name]
        # 從zip文件中直接讀取圖片
        with zip_ref.open(filename) as image_file:
            image = Image.open(image_file).convert('RGB')
        if self.transform:
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
train_data = HandWrite(zip_files, worddict, transform=transform, end_ratio=0.8)
test_data = HandWrite(zip_files, worddict, transform=transform, start_ratio=0.8)

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