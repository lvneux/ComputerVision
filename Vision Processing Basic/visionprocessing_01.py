import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

t, bin_img = cv.threshold(gray[:,:], 127, 255, cv.THRESH_BINARY)

h=cv.calcHist([bin_img],[0],None,[256],[0,256])
h2=h=cv.calcHist([gray],[0],None,[256],[0,256])
plt.imshow(bin_img,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()
plt.plot(h,color='r',linewidth=1)
plt.plot(h2,color='r',linewidth=1)
plt.show()
