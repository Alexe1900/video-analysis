import os
import cv2

isClass = open('./classified.txt', 'r')
classified = dict()

classifiedN = int(isClass.readline().strip())

for _ in range(classifiedN):
    classified[isClass.readline().strip()] = 1

isClass.close()

directory = './DATASETS/ownHanoi/images'

results = open('./classes.txt', 'a')

try:
    for file in os.scandir(directory):
        if file.is_file():
            imgName = file.path.split('\\')[-1]
            if imgName in classified:
                continue
            image = cv2.imread(file.path)
            cv2.imshow(imgName, image)
            cv2.waitKey(1)

            result = input('enter classes:')
            results.write(f'{imgName} {result}\n')
            classified[imgName] = 1
            
            cv2.destroyAllWindows()
except KeyboardInterrupt:
    print('interrupted')
    isClass = open('./classified.txt', 'w')
    filelist = [str(len(classified))]
    for key in classified.keys():
        filelist.append(key)
    isClass.write('\n'.join(filelist))
    isClass.close()

results.close()