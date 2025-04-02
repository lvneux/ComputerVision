import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

img1 = cv.imread('./imgs/mot_color70.jpg')[190:350,440:560]
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./imgs/mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수 : ', len(kp1), len(kp2))

start = time.time()

#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#search_params = dict(checks=50)

#flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
#matches = flann_matcher.knnMatch(des1, des2, 2)

bf_matcher = cv.BFMatcher_create(cv.NORM_L2)
matches = bf_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
print('매칭에 걸린 시간 : ', time.time()-start)

img_match = np.empty((max(img1.shape[0], img2.shape[0]), 
                      img1.shape[1]+img2.shape[1],3),dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_match = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.imshow(img_match)
plt.axis('off') 
plt.title('Good Matches')
plt.show()
