import shutil
import os
import random

names = os.listdir('./DATASETS/ownHanoi/images')

distrib = list(range(len(names)))
random.shuffle(distrib)

n = len(distrib)

for i in range(730):
    shutil.copy('./DATASETS/ownHanoi/images/' + names[distrib[i]],
                './DATASETS/ownSplit/train/' + names[distrib[i]])

for i in range(730, 913):
    shutil.copy('./DATASETS/ownHanoi/images/' + names[distrib[i]],
                './DATASETS/ownSplit/val/' + names[distrib[i]])