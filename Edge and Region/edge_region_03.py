import skimage
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

src = skimage.data.coffee()

mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

iterCount = 1
mode = cv.GC_INIT_WITH_RECT

rc = cv.selectROI(src)

cv.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask==0) | (mask==2), 0, 1).astype('uint8')
dst = src*mask2[:,:,np.newaxis]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(src)
axs[0].set_title('Original Image')
axs[1].imshow(mask2, cmap='gray')
axs[1].set_title('Mask')
axs[2].imshow(dst)
axs[2].set_title('Object Extracted')
plt.show()

def on_mouse(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(dst, (x,y), 3, (255,0,0), -1)
        cv.circle(mask, (x,y), 3, cv.GC_FGD, -1)
        cv.imshow('dst', dst)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(dst, (x,y), 3, (0,0,255), -1)
        cv.circle(mask, (x,y), 3, cv.GC_BGD, -1)
        cv.imshow('dst', dst)    
    elif event==cv.EVENT_MOUSEMOVE:
        if flags&cv.EVENT_FLAG_LBUTTON:
            cv.circle(dst, (x,y), 3, (255,0,0), -1)
            cv.circle(mask, (x,y), 3, cv.GC_FGD, -1)
            cv.imshow('dst', dst)           
        elif flags&cv.EVENT_FLAG_RBUTTON:
            cv.circle(dst, (x,y), 3, (0,0,255), -1)
            cv.circle(mask, (x,y), 3, cv.GC_BGD, -1)
            cv.imshow('dst', dst)            

cv.setMouseCallback('dst', on_mouse)

while True: 
    key=cv.waitKey()
    if key == 13:
        cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
        np.where((mask==2) | (mask==0), 0, 1).astype('uint8')     
        dst = src*mask2[:,:,np.newaxis]
        cv.imshow('dst',dst)
    elif key == 27:
        break

cv.destroyAllWindows()
        