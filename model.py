import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as trfs

device = torch.device("cpu")
if (torch.cuda.is_available()):
    print('cuda available')
    device = torch.device("cuda")

transform = trfs.Compose([
    trfs.ToTensor(),
    trfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    trfs.Resize((640, 640))
])