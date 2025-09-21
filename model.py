import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as trfs
from torch.utils.data import Dataset

import pandas as pd
import os

device = torch.device("cpu")
if (torch.cuda.is_available()):
    print('cuda available')
    device = torch.device("cuda")

transform = trfs.Compose([
    trfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    trfs.Resize((128, 128))
])

class ImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = tv.io.decode_image(img_path)
        image = image.float() / 255.0
        if self.transform:
            image = self.transform(image)
        
        strLabel = str(self.img_labels.iloc[idx, 1])
        while (len(strLabel) != 7):
            strLabel += '0'

        listLabel = [int(c) for c in strLabel]
        label = torch.tensor(listLabel)

        return image, label

train = ImgDataset('classes.csv', './DATASETS/ownSplit/train', transform)
val = ImgDataset('classes.csv', './DATASETS/ownSplit/val', transform)