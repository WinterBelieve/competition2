import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
import zipfile
import os

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

# define a function to get the label from the filename
def get_label(filename):
    return filename.split('/')[-1].split('_')[0]

# get all the files
files = glob.glob(f'{dataset_dir}/*.png')

# map label to index
worddict = {}
for f in files:
    label = get_label(f)
    if label not in worddict:
        worddict[label] = len(worddict)

# define the dataset class
class HandWrite(Dataset):
    def __init__(self, files, worddict, transform=None, start=0, end=0.9):
        self.worddict = worddict
        files = list(filter(lambda x: get_label(x) in worddict, files))
        self.files = files[int(len(files) * start):int(len(files) * end)]
        if transform:
            self.transform = transforms.Compose([transforms.ToTensor()] + transform)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def get_image(self, fname):
        return self.transform(Image.open(fname))

    def get_label(self, fname):
        return self.worddict[get_label(fname)]

    def __getitem__(self, idx):
        return self.get_image(self.files[idx]), self.get_label(self.files[idx])

# create train, test, and validation datasets
batch_size = 32
ch_traindata = HandWrite(files, worddict, None, 0, 0.8)
ch_testdata = HandWrite(files, worddict, None, 0.8, 0.9)
ch_validdata = HandWrite(files, worddict, None, 0.9, 1.0)

# create DataLoader
ch_trainds = DataLoader(ch_traindata, batch_size=batch_size, num_workers=2, shuffle=True)
ch_testds = DataLoader(ch_testdata, batch_size=batch_size, num_workers=2, shuffle=False)
ch_validds = DataLoader(ch_validdata, batch_size=batch_size, num_workers=2, shuffle=False)

# Define the model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(2352, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
    torch.nn.LogSoftmax(dim=1)
)
# define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 28 * 28, 128),  # 假设图片是28x28像素的彩色图片
    nn.ReLU(),
    nn.Linear(128, len(worddict)),  # 输出层的大小匹配类别的数量
    nn.LogSoftmax(dim=1)
)

# define the loss function and the optimizer
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
EPOCHS = 10
for ep in range(EPOCHS):
    model.train()
    for images, labels in ch_trainds:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {ep+1}, Loss: {loss.item()}')

# save the model
torch.save(model.state_dict(), 'handwrite_model.pth')

print("Model trained and saved successfully.")