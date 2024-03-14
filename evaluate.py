import torch
import torchvision.transforms as transforms
import json
from torchvision import datasets
import sys, os
import zipfile

def evaluate_model(model):
    # Evaluate the model
    # Define the dataset directory and zip files
    base_dir = 'Traditional-Chinese-Handwriting-Dataset/data'
    dataset_dir = os.path.join(base_dir, 'cleaned_data(50_50)')
    zip_files = [
        'cleaned_data(50_50)-20200420T071507Z-001.zip',
        'cleaned_data(50_50)-20200420T071507Z-002.zip',
        'cleaned_data(50_50)-20200420T071507Z-003.zip',
        'cleaned_data(50_50)-20200420T071507Z-004.zip'
    ]

    # Check and extract zip files
    for zip_file in zip_files:
        zip_path = os.path.join(base_dir, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Evaluate the model on test dataset
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
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

# Load model from sys.argv[1]
model_path = sys.argv[1]  # Assuming the first argument is the model path
model = torch.load(model_path)
evaluate_model(model)
