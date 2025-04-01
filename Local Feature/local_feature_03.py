import cv2 as cv
import numpy as np

img1 = cv.imread('./imgs/img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./imgs/img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf_matcher = cv.BFMatcher_create(cv.NORM_L2)
matches = bf_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)

if len(good_match) > 4:  

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_match])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_match])

    H, mask = cv.findHomography(points1, points2, cv.RANSAC)

    h, w = img2.shape[0], img2.shape[1]
    img1_warped = cv.warpPerspective(img1, H, (w, h))

    imgs = np.hstack((img1_warped, img1))
    
    comparison = cv.addWeighted(img1_warped, 0.5, img2, 0.5, 0)

    cv.imshow('Warped Image', imgs)
    cv.imshow('Comparison', comparison)

    img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Feature Matches', img_match)

    cv.waitKey(0)
    cv.destroyAllWindows()
