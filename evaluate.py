import torch
import torchvision.transforms as transforms
import json
from torchvision import datasets
import sys, os
import zipfile

def evaluate_model(model):
    # Evaluate the model
    # Load test dataset
    zip_files = [
        'cleaned_data(50_50)-20200420T071507Z-001.zip',
        'cleaned_data(50_50)-20200420T071507Z-002.zip',
        'cleaned_data(50_50)-20200420T071507Z-003.zip',
        'cleaned_data(50_50)-20200420T071507Z-004.zip'
    ]
    dataset_dir = 'Traditional-Chinese-Handwriting-Dataset/data'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract files without overwriting
                for member in zip_ref.infolist():
                    filename = member.filename
                    # Adjust filename to reflect new structure
                    adjusted_filename = os.path.join(dataset_dir, filename.split('/', 1)[-1])
                    if not os.path.exists(os.path.dirname(adjusted_filename)):
                        os.makedirs(os.path.dirname(adjusted_filename))
                    zip_ref.extract(member, os.path.dirname(adjusted_filename))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Iterate all batches on the dataset
    correct = 0
    total = 0
    num = 0
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        num += labels.size(0)
        if sys.argv[2] != 'testing':
            if num > len(test_dataset) * 0.2:
                break
        correct += (predicted == labels).sum().item()

    # Return evaluation result
    ret = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "status": "success %g" % (correct / total * 100)
    }
    # Return json string
    print(json.dumps(ret))

# Load model from argv[1]
model = torch.load(sys.argv[1])
evaluate_model(model)
