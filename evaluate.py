import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch import nn
import sys,os
from PIL import Image
import json

from torch.utils.data.dataloader import default_collate

class SimpleDataset(Dataset):
    def __init__(self, directory, classmap, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.classmap = classmap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        l = image_path.split('/')[-1].split('.')[0]

        if l in self.classmap:
            label = self.classmap[l]
            return image, label  # normal return
        else:
            print(f"Label for {l} not found in classmap. Skipping this item.")
            return None  # skip this item


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
# print(dataset.class_to_idx) # class to index mapping

def custom_collate_fn(batch):
    # filter out Nones
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()  # return empty tensors
    return default_collate(batch)

test_dataset = torch.utils.data.Subset(dataset, range(len(dataset)-200,len(dataset)))
test_loader = DataLoader(test_dataset, batch_size=128)
classmap = json.loads(open('class_to_idx.txt').read())
# print(classmap) # class to index mapping

# model_path = '/home/jovyan/competition2/resnet_ChMNIST.pth'
# model_path = '/home/jovyan/competition2/handwrite_model.pth'
if sys.argv[2] != 'training':
     testds = SimpleDataset('newchinese', classmap,transform=transform)
     test_loader = DataLoader(testds, batch_size=128, collate_fn=custom_collate_fn)
model_path = sys.argv[1]
model = torch.jit.load(model_path).to(device)

# Function to evaluate the model and identify misclassifications
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    misclassified = []  # List to store misclassified examples

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Handle multiple predictions and labels in a batch
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    label_name = dataset.classes[labels[i].item()]
                    predicted_name = dataset.classes[predicted[i].item()]
                    misclassified.append((label_name, predicted_name))

    accuracy = 100 * correct / total
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

    # Print misclassified examples
    print("Misclassified examples:")
    for label, pred in misclassified:
        print(f"Actual: {label}, Predicted: {pred}")

# Evaluate the model
evaluate_model(model, test_loader)
