import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data
import zipfile
import os
import numpy as np

# zip files
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

# define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# read dataset
full_ds = datasets.ImageFolder(root=dataset_dir, transform=transform)

# split dataset
train_size = int(0.8 * len(full_ds))
test_size = len(full_ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

# Get DataLoader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

# Define the model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(2352, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
    torch.nn.LogSoftmax(dim=1)
)

# 定義loss funtion和optimizer
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
EPOCHS = 10
for ep in range(EPOCHS):
    model.train()
    tot_loss = 0
    tot_success = 0
    count = 0
    for images, labels in train_dl:
        images = images.view(images.shape[0], -1)  # flatten the images
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        success = (torch.argmax(outputs, dim=1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_success += success
        count += len(images)
    print(f'Epoch {ep}, Loss: {tot_loss / count}, Accuracy: {tot_success / count * 100}%')

# Evaluate the model
torch.save(model, 'model.pth')
