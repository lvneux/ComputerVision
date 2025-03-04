import cv2 as cv
import numpy as np
import sys

img = cv.imread('./sample.jpg')

if img is None:
    sys.exit('File not found')
    
start_x, start_y, end_x, end_y, crop_img = None, None, None, None, None
drawing = False

def draw(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, crop_img, drawing, img
    
    temp = img.copy()
    
    if event==cv.EVENT_LBUTTONDOWN:
        start_x,start_y = x,y
        drawing = True
        
    elif event==cv.EVENT_MOUSEMOVE and drawing:
        cv.rectangle(temp,(start_x, start_y),(x, y),(0,0,255),2)
        cv.imshow('Drawing', temp)
        
    elif event==cv.EVENT_LBUTTONUP:
        end_x,end_y = x,y
        drawing = False
        crop_img = img[start_y:end_y, start_x:end_x]
        cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,0,255),2)
        cv.imshow('Drawing', img)
        cv.imshow('Cropped', crop_img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
    elif cv.waitKey(1)==ord('r'):
        img = cv.imread('./sample.jpg')
        cv.imshow('Drawing', img)
    elif cv.waitKey(1)==ord('s'):
        cv.imwrite('crop_img.jpg', crop_img)
        
cv.destroyAllWindows()
