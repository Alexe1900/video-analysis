import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision as tv
import torchvision.transforms.functional as trff
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import pandas as pd
import os

device = torch.device("cpu")
if (torch.cuda.is_available()):
    print('cuda available')
    device = torch.device("cuda")

class ImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = tv.io.decode_image(img_path).float() / 255.0
        if image.shape[0] == 4:
            image = image[:3, :, :]
        
        image = trff.resize(image, [128, 128])
        image = (image - 0.5) / 0.5
        
        strLabel = str(self.img_labels.iloc[idx, 1])
        while (len(strLabel) != 7):
            strLabel += '0'

        listLabel = [int(c) for c in strLabel]
        label = torch.tensor(listLabel)

        return image, label

trainSet = ImgDataset('./DATASETS/ownSplit/labels/train.csv', './DATASETS/ownSplit/train')
valSet = ImgDataset('./DATASETS/ownSplit/labels/val.csv', './DATASETS/ownSplit/val')

trainLoad = DataLoader(
    trainSet,
    32,
    True
)

valLoad = DataLoader(
    valSet,
    32,
    True
)

print('DATASET LOADED')

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(16 * 16 * 128, 512)
        self.fc2 = nn.Linear(512, 28)
    
    def forward(self, x):
        x = self.pool(nnf.relu(self.conv1(x)))
        x = self.pool(nnf.relu(self.conv2(x)))
        x = self.pool(nnf.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))

        x = x.view(-1, 4, 7)
        x = nnf.softmax(x, 2)

        return x

net = NeuralNet()
net.to(device)

lossF = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    runningLoss = 0.0

    for i, data in enumerate(trainLoad):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossF(outputs, labels)

        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

        if (i % 10 == 9):
            print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {runningLoss / 10:.3f}')
            runningLoss = 0.0
        else: print(f'[{epoch + 1}/{epochs}, {i + 1:5d}]')