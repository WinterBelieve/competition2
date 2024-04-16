import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import json
import sys

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
model_path = sys.argv[1]
model = torch.jit.load(model_path).to('cpu')

# model evaluation
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# evaluate model
evaluate_model(model, test_loader)
