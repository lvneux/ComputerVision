import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(bin_img,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

img_dilate = cv.morphologyEx(b, cv.MORPH_DILATE, kernel)
img_erode = cv.morphologyEx(b, cv.MORPH_ERODE, kernel)
img_open = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)
img_close = cv.morphologyEx(b, cv.MORPH_CLOSE, kernel)

imgs = np.hstack((b,img_dilate,img_erode,img_open,img_close))

cv.imshow('Original - Dilate - Eroded - Opened - Closed', imgs)

cv.waitKey()
cv.destroyAllWindows()
