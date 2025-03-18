import cv2 as cv
import sys
import numpy as np

img = cv.imread('./tree.png')

if  img is None:
    sys.exit('파일을 찾을 수 없습니다.')

rows, cols = img.shape[:2]

rot = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1.5) 
dst = cv.warpAffine(img, rot, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)

start_x = (dst.shape[1] - cols) // 2
start_y = (dst.shape[0] - rows) // 2

dst_crop = dst[start_y:start_y + rows, start_x:start_x + cols]

imgs = np.hstack((img, dst_crop))

cv.imshow('Result', imgs)
cv.waitKey()

cv.destroyAllWindows()
