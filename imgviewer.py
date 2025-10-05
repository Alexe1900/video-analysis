import cv2 as cv

img = cv.imread('./DATASETS/ownSplit/train/img6-645-_png.rf.391c295797c6fa3ac09c22759cbf0750.jpg')
resImage = cv.resize(img, (128, 128))
cv.imshow('blub', resImage)
cv.waitKey()

cv.destroyAllWindows()