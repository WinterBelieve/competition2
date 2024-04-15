import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json
import sys

# check if model path is provided
if len(sys.argv) < 2:
    print("Usage: python evaluate.py <model_path>")
    sys.exit(1)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # resize
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

# data
data_dir = 'Traditional-Chinese-Handwriting-Dataset/data/cleaned_data(50_50)'
test_data = datasets.ImageFolder(root=data_dir, transform=transform)

# if you want to split the data into train and test
_, test_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.8), len(test_data)-int(len(test_data)*0.8)])

test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

# model path
model_path = sys.argv[1]  
model = torch.jit.load(model_path).to(device)

# model evaluation
def evaluate_model(model, test_loader):
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # no need to calculate gradients
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

# evaluate model
evaluate_model(model, test_loader)
