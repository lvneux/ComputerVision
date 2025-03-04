import cv2 as cv
import numpy as np
import sys

img = cv.imread('./sample.jpg')

if img is None:
    sys.exit('File not found')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

imgs=np.hstack((img, gray_img))
        
cv.imshow('Collected Images', imgs)
    
cv.waitKey()
cv.destroyAllWindows()