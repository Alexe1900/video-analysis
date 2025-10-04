import os

labels = {}

inp = open('classes.csv', 'r')
inpList = inp.read().split('\n')

tOut = open('./DATASETS/ownSplit/labels/train.csv', 'a')
vOut = open('./DATASETS/ownSplit/labels/val.csv', 'a')

for lPair in inpList:
    lpSplit = lPair.split(',')
    labels[lpSplit[0]] = lpSplit[1]

for tImg in os.listdir('./DATASETS/ownSplit/train/'):
    tOut.write(tImg + ',' + labels[tImg] + '\n')

for vImg in os.listdir('./DATASETS/ownSplit/val/'):
    vOut.write(vImg + ',' + labels[vImg] + '\n')