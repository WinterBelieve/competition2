import torch
from torchvision import transforms, datasets
import torch.utils.data
import json
import sys
import os
import zipfile
from torch import nn

# 定义 CustomDataset 以适应特定的数据结构
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def evaluate_model(model, dataset_dir, transform):
    # Load test dataset using CustomDataset
    test_dataset = CustomDataset(root=dataset_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return evaluation result
    ret = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "status": f"success {correct / total * 100}%"
    }
    print(json.dumps(ret))

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Assuming images are to be resized to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For colored images
])

# Assuming the first argument is the model path
model_path = sys.argv[1]
model = torch.load(model_path, map_location=torch.device('cpu'))

# Define the dataset directory
base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
dataset_dir = os.path.join(base_dir, 'cleaned_data(50_50)')

# Check and extract zip files if necessary
zip_files = [
    'cleaned_data(50_50)-20200420T071507Z-001.zip',
    'cleaned_data(50_50)-20200420T071507Z-002.zip',
    'cleaned_data(50_50)-20200420T071507Z-003.zip',
    'cleaned_data(50_50)-20200420T071507Z-004.zip'
]
for zip_file in zip_files:
    zip_path = os.path.join(base_dir, zip_file)
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

evaluate_model(model, dataset_dir, transform)
