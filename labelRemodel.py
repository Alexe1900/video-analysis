import os

classes = open('./classes.txt', 'r')
newclasses = open('./newclasses.txt', 'a')

labels = classes.read().split('\n')
newLabels = []

for label in labels:
    newLabels.append([])
    newLabels[-1].append(label.split(' ')[0])
    newLabels[-1].append(''.join(label.split(' ')[1:]))
    newclasses.write(' '.join(newLabels[-1])+'\n')