import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp,des = sift.detectAndCompute(gray,None)

gray = cv.drawKeypoints(gray,kp,None,flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
axs[1].set_title('SIFT')
axs[1].axis('off')

plt.show()
