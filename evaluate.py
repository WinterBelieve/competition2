import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import json
import sys

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # resize
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

# data loader
data_dir = 'Traditional-Chinese-Handwriting-Dataset/data/cleaned_data(50_50)'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
_, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
test_loader = DataLoader(test_dataset, batch_size=128)

# model path
model_path = 'handwrite_model.pth'
if len(sys.argv) > 1:
    model_path = sys.argv[1]

model = torch.jit.load(model_path).to(device)

# model evaluation
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# evaluate model
evaluate_model(model, test_loader)
