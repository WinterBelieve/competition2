import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data
import zipfile
import os

# 解壓縮資料集
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
        # 檢查是否已經有相同名字的檔案夾存在
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        # 解壓縮每個檔案，確保不覆蓋已存在的檔案
        zip_ref.extractall(base_dir)

# 定義轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 讀取資料集
ds = datasets.ImageFolder(root=dataset_dir, transform=transform)

# 獲取 dataloader
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

# 定義簡單模型
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),  # 注意根據你的實際圖片大小調整
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),  # 輸出層數量應該與你的類別數相匹配
    torch.nn.LogSoftmax(dim=1)
)

# 定義損失函數和優化器
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
EPOCHS = 10
for ep in range(EPOCHS):
    tot_loss = 0
    tot_success = 0
    count = 0
    for images, labels in dl:
        # 調整 images 的大小
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        success = (torch.argmax(outputs, dim=1) == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        tot_success += success
        count += len(images)
    print(f'Epoch {ep}, Loss: {tot_loss/count}, Accuracy: {tot_success/count*100}%')

# 保存模型
torch.save(model, 'model.pth')