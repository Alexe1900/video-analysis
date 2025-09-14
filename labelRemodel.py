classes = open('./classes.txt', 'r')
newClasses = open('./newclasses.txt', 'a')

labels = classes.read().split('\n')
newLabels = []

for label in labels:
    newLabels.append([])
    newLabels[-1].append(label.split(' ')[0])

    newLabels[-1].append(label.split(' ')[1])
    newClass = ''
    for i in newLabels[-1][1]:
        newClass += str(int(i) + 1)
    newLabels[-1][1] = newClass
    
    newClasses.write(' '.join(newLabels[-1])+'\n')