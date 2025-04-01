import cv2 as cv
import numpy as np

img1 = cv.imread('img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf_matcher = cv.BFMatcher_create(cv.NORM_L2)
matches = bf_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_matches = [m for m, n in matches if m.distance < T * n.distance]

if len(good_matches) > 4:  

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    H, mask = cv.findHomography(points1, points2, cv.RANSAC)

    h, w = img2.shape[0], img2.shape[1]
    img1_warped = cv.warpPerspective(img1, H, (w, h))

    side_by_side = np.hstack((img1_warped, img1))
    
    comparison = cv.addWeighted(img1_warped, 0.5, img2, 0.5, 0)

    cv.imshow('Warped Image', side_by_side)
    cv.imshow('Comparison', comparison)

    img_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Feature Matches', img_match)

    cv.waitKey(0)
    cv.destroyAllWindows()