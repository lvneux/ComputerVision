import cv2 as cv
import sys

img = cv.imread('./tree.png')

if  img is None:
    sys.exit('파일을 찾을 수 없습니다.')

rows, cols = img.shape[:2]

rot = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1.5) 
dst = cv.warpAffine(img, rot, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)

cv.imshow('src', img)
cv.imshow('dst', dst)
cv.waitKey()

cv.destroyAllWindows()
