import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened:
    sys.exit('Camera connection failed')

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Break')
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    edges = cv.Canny(gray, 100, 200)
    
    gray_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    
    result=np.hstack((frame, gray_edges))
        
    cv.imshow('Video display', result)
    
    key = cv.waitKey(1)
    if key==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
