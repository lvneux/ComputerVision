import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./edgeDetectionImage.jpg')

if  img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 100, 200)

lines = cv.HoughLinesP(canny, 1, np.pi / 180., 100, minLineLength=35, maxLineGap=5)

drawn_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cv.cvtColor(drawn_img, cv.COLOR_BGR2RGB))
axs[1].set_title('Detected Lines')
axs[1].axis('off')

plt.show()
