import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 將圖像大小調整為 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 標準化
])

# 數據加載
data_dir = 'Traditional-Chinese-Handwriting-Dataset/data/cleaned_data(50_50)'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)

# 模型架構
num_classes = len(dataset.classes)  # 根據數據集類別數量設置
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 64 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes),
    nn.LogSoftmax(dim=1)
)

# 損失函數和優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
EPOCHS = 1
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / total}, Accuracy: {100 * correct / total}%')

# 評估模型（可選）
# 這裡可以添加測試集上的評估代碼
model_scripted = torch.jit.script(model)
model_scripted.save('handwrite_model.pth')
