import cv2 as cv
import numpy as np
import sys

img = cv.imread('./soccer.jpg')

if img is None:
    sys.exit('File not found')

img_2 = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)

gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

imgs=np.hstack((img_2, gray_img))
        
cv.imshow('Collected Images', imgs)
    
cv.waitKey()
cv.destroyAllWindows()
