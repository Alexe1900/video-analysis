classesTxt = open('classes.txt', 'r')
classesCsv = open('classes.csv', 'a')

classes = classesTxt.read().split('\n')

for cls in classes:
    classesCsv.write(','.join(cls.split(' ')) + '\n')