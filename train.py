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

# 解壓縮後資料的根目錄
root_dir = 'Traditional-Chinese-Handwriting-Dataset/data'

# 資料夾名稱固定為 'cleaned_data(50_50)'
dataset_dir = os.path.join(root_dir, 'cleaned_data(50_50)')

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(root_dir)

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
    torch.nn.Linear(784, 128),  # 圖片大小需調整為實際大小
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),  # 輸出層的數量應該匹配類別數量
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