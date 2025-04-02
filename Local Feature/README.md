# 01. SIFT를 이용한 특징점 검출 및 시각화

## 과제 설명 및 요구사항
  + 설명
     + 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용해 특징점 검출
   
  + 요구사항
      + cv.imread()를 사용하여 이미지 로드
      + cv.SIFT_create()를 사용하여 SIFT 객체 생성
      + detectAndCompute()를 사용하여 특징점 검출
      + cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
      + matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력
        
## 전체 코드 
   ```
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./imgs/mot_color70.jpg')
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
 ```

## 이미지 로드 및 그레이스케일 변환
 ```
img = cv.imread('./imgs/mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 ```

## SIFT 객체 생성, 특징점 검출, 특징점 시각화
 ```
sift = cv.SIFT_create()
kp,des = sift.detectAndCompute(gray,None)
gray = cv.drawKeypoints(gray,kp,None,flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
 ```
+ SIFT(Scale Invariant Feature Transform)
  +  이미지의 크기와 회전에 불변하는 특징을 추출하는 알고리즘
+ cv.SIFT_create(nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None)
  + nfeatures : 보유할 최대 특징점 수. 0일 경우 모든 특징점을 유지(default=0)
  + nOctaveLayers  : 이미지 피라미드 계층(옥타브) 당 레이어 수. 값 증가 시 다양한 스케일 탐지가 가능하지만 연산 비용 증가(default=3)
  + contrastThreshold : 낮은 대비 영역의 특징점 필터링. 값 증가 시 검출되는 특징점 수 감소(default=0.04)
  + edgeThreshold : 에지 영역의 불안정한 특징점 필터링. 값 증가 시 더 많은 에지 특징점 허용(default=10)
  + sigma : 첫 번째 옥타브에서 사용되는 가우시안 필터의 표준 편차. 이미지 스무딩 정도 제어(default=1.6)
+ detectAndCompute(image, mask=None)
  + 이미지에서 특징점과 Descriptors를 계산하는 함수
  + image : 처리할 입력 이미지(grayscale 권장)
  + mask : 특정 영역만 처리할 때 사용하는 마스크 
  + kp
    + 검출된 특징점 리스트
    + 각 특징점은 위치(x,y), 특징점 영역 반지름, 방향(radian), 특징점 강도 정보를 포함
  + des
    + descriptor 배열(128차원 벡터로 구성된 numpy 배열)
    + 각 특징점 주변의 국소 패턴을 설명하는 고유한 벡터 
+ cv.drawKeypoints(image, keypoints, outImage=None, color=None, flags=None) -> outImage
  + 검출된 특징점을 시각화하는 유틸리티 함수
  + image : 원본 입력 이미지
  + keypoints : 검출된 특징점 정보
  + outImage : 출력 이미지
  + color : 특징점 표현 색상으로, default의 경우 임의의 색상으로 표현(default=-1,-1,-1,-1)
  + flags
    + 특징점 표현 방법
    + DEFAULT : 중심점만 표시(크기/방향 표시하지 않음)
    + DRAW_RICH_KEYPOINTS : 크기와 방향을 포함한 풍부한 시각화
    + NOT_DRAW_SINGLE_POINTS : 단일 점 표시 안함
 + 이미지에서 특징점 원의 크기가 다른 이유 : cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS 플래그에 의해서 원의 크기는 특징점이 검출된 스케일 크기의 영향을 받으며, 검출된 스케일에 비례하는 크기의 원이 그려짐(원이 클수록 해당 특징점은 더 큰 영역에서 의미 있는 패턴을 가지고 있으며, 작은 원은 더 세밀한 특징을 나타낸다고 해석할 수 있음)

## matplotlib을 사용한 결과 시각화
 ```
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
axs[1].set_title('SIFT')
axs[1].axis('off')

plt.show()
 ```

## 실행 결과 1 - 기본 요구사항
   <img src="https://github.com/user-attachments/assets/8d0ca90c-33eb-43cf-b569-824af4911b76"/>

## 실행 결과 2 - SIFT_create()의 parameter 변경에 따른 결과 비교 
   <img src="https://github.com/user-attachments/assets/d91a14bd-d290-4d4f-aa11-55d0cb77e039"/>
   
# 02. SIFT를 이용한 두 영상 간 특징점 매칭

## 과제 설명 및 요구사항
  + 설명
     + 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과 시각화
   
  + 요구사항
      + cv.imread()를 사용하여 두 개의 이미지 로드
      + cv.SIFT_create()를 사용하여 특징점 추출
      + cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점 매칭
      + cv.drawMatches()를 사용하여 매칭 결과 시각화
      + matplotlib을 이용하여 매칭 결과 출력
        
## 전체 코드 
   ```
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
 ```

## 이미지 로드 및 그레이스케일 변환
 ```
img1 = cv.imread('./imgs/mot_color70.jpg')[190:350,440:560]
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./imgs/mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
 ```
+ cv.imread('./imgs/mot_color70.jpg')[190:350,440:560]를 사용해 버스를 크롭하여 모델 영상으로 사용

## SIFT 특징점 추출
 ```
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수 : ', len(kp1), len(kp2))
 ```

## cv.FlannBasedMatcher() - FLANN Matcher 초기화
 ```
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
 ```
+ cv.FlannBasedMatcher(index_params, search_params)
  + index_params : Flann 기반 매칭에서 인덱스를 설정하는 파라미터로, 딕셔너리 형태로 전달(하위 parameter : algorithm, trees)
    + algorithm : 매칭에 사용할 알고리즘
      + 0 : Linear, 모든 데이터 타입에 적합
      + 1 : KD-Tree, float 데이터 타입에 적합
      + 6 : LSH, binary 데이터 타입에 적합
    + trees : 매칭에 사용될 트리의 개수
  + search_params : 매칭을 수행할 때 검색할 이웃의 개수와 검색할 최대 거리를 설정하는 파라미터로, 딕셔너리 형태로 전달(하위 parameter : checks, eps)
    + checks : 검색할 이웃의 개수
    + eps : 검색할 최대 거리

## KNN 매칭
 ```
knn_match = flann_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
 ```
+ knnMatch(queryDescriptors, trainDescriptors, k, mask=None, compactResult=None)
  + queryDescriptors : 각 특징점에 대한 descriptor(비교할 첫 번째 이미지)
  + trainDescriptors : 비교 대상이 되는 descriptor
  + k : 반환할 최근접 이웃 수 (k는 1보다 같거나 커야함)
  + mask : 허용 가능한 매칭 마스크 (query × train 크기, default=None)
  + compactResult : 매칭 결과 압축 여부. False면 매칭이 없는 경우에도 결과가 포함되고, True면 매칭이 없는 특징점은 생략되고 k개로 제한된 매칭 결과만 포함(default=False)
+ 각 descriptors의 상위 2개 매칭 결과 반환
+ distance ratio < 0.7인 경우만 유효 매칭(good_match)으로 설정 (distance ratio: nearest1과 nearest2의 거리 비율로, 낮을수록 고유성이 높음)

## cv.drawMatches() - 매칭 결과 시각화
 ```
img_match = np.empty((max(img1.shape[0], img2.shape[0]), 
                      img1.shape[1]+img2.shape[1],3),dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_match = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)
 ```
+ np.empty() : 두 이미지를 나란히 배치할 수 있는 새로운 이미지 생성(높이=max(h1,h2), 너비=w1+w2)
+ cv.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor=None, singlePointColor=None, matchesFlags=None)
  + img1 : 첫 번째 이미지로, 특징점 keypoint1이 추출된 이미지
  + keypoint1 : 첫 번째 이미지의 특징점 리스트(각 특징점의 위치 및 특성 포함)
  + img2 : 두 번째 이미지로, 특징점 keypoint2가 추출된 이미지
  + keypoint2 : 두 번째 이미지의 특징점 리스트(각 특징점의 위치 및 특성 포함)
  + matches1to2 :매칭된 특징점 쌍들의 리스트(각 매칭은 keypoints1의 특징점과 keypoints2의 특징점 간의 매칭을 나타)
  + outImg : 출력 이미지로, 두 이미지를 결합하여 매칭된 특징점들을 그린 결과 이미지를 저장할 배열
  + matchColor : 매칭된 선을 그릴 색상. BGR 형식(default=(0,255,0))
  + singlePointColor : 특징점을 나타내는 점의 색상. BGR 형식(default=(255,0,0))
  + matchesFlags : 매칭을 그릴 때 사용하는 플래그
    + cv.DrawMatchesFlags_DEFAULT : 기본적인 매칭 표시(default값)
    + cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS : 단일 점을 그리지 않음
    + cv.DrawMatchesFlags_DRAW_OVER_OUTIMG : 결과 이미지에 직접 그려서 표시
+ img1과 img2의 특징점을 시각적으로 연결

## matplotlib을 사용한 결과 시각화
 ```
plt.figure(figsize=(12, 6))
plt.imshow(img_match)
plt.axis('off') 
plt.title('Good Matches')
plt.show()
 ```

## 실행 결과
   <img src="https://github.com/user-attachments/assets/237103b4-2190-4d02-bc06-e34da4beb38c"/>

# 03. 호모그래피를 이용한 이미지 정합(Image Alignment)

## 과제 설명 및 요구사항
  + 설명
     + SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬
     + 샘플 파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개 선택

  + 요구사항
      + cv.imread()를 사용하여 두 개의 이미지 로드
      + cv.SIFT_create()를 사용하여 특징점 검출
      + cv.BFMatcher()를 사용하여 특징점 매칭
      + cv.findHomography()를 사용하여 호모그래피 행렬 계산
      + cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
      + 변환된 이미지를 원본 이미지와 비교하여 출력
        
## 전체 코드 
   ```
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
 ```

## 이미지 로드 및 그레이스케일 변환
 ```
img1 = cv.imread('./imgs/img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./imgs/img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
 ```

## SIFT 특징점 추출 
 ```
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
 ```

## cv.BFMatcher() - 특징점 매칭
 ```
bf_matcher = cv.BFMatcher_create(cv.NORM_L2)
matches = bf_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
 ```
+ cv2.BFMatcher_create(normType=None, crossCheck=None)
  + normType : 거리 측정 알고리즘 지정(default=cv.NORM_L2)
    + cv.NORM_L1 : L1 norm 사용
    + cv.NORM_L2 : L2 norm 사용
    + cv.NORM_HAMMING : 해밍 거리 사용
    + cv.NORM_HAMMING2 : 두 비트를 한 단위로 취급하여 해밍 거리 계산
  + crossCheck : boolean 타입으로, True면 양방향 매칭 결과가 같은 경우만 반환(default=False)

## cv.findHomography() - 호모그래피 행렬 계산 
 ```
if len(good_match) > 4:  

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_match])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_match])

    H, mask = cv.findHomography(points1, points2, cv.RANSAC)
 ```
+ 호모그래피 추정을 위해서는 최소 4개의 매칭 쌍이 있어야하므로, <b>len(good_match) > 4</b>를 사용해 조건 제
+ point1, point2 좌표 추출
  + queryIdx
    + 첫 번째 이미지의 키포인트 인덱스
    + 예를 들어 kp1[5].pt인 경우, kp1[5]의 값 (x1, y1)은 img1의 5번째 키포인트 위치를 의미함
  + trainIdx
    + 두 번째 이미지의 대응 키포인트 인덱스
    + 예를 들어 kp1[5].pt인 경우, kp2[5]의 값 (x2, y2)는 img2의 5번째 키포인트 위치를 의미함
+ cv.findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, mask=None, maxIters=None, confidence=None) -> retval, mask
  + srcPoints : 1번 이미지 특징점 좌표
  + dstPoints : 2번 이미지 특징점 좌표
  + method : 호모그래피 행렬 계산 방법. 0, LMEDS, RANSAC, RHO 중 선택(defaule=0, 이상치가 있을 경우=RANSAC,RHO 권장)
  + ansacReprojThreshold : RANSAC 재투영 에러 허용치(default=3)
  + maxIters : RANSAC 최대 반복 횟수(default=2000)
  + retval: 호모그래피 행렬
  + mask: 출력 마스크 행렬(RANSAC, RHO 방법 사용 시 Inlier로 사용된 점들을 1로 표시한 행렬)

## 이미지 변환 및 시각화
```
    h, w = img2.shape[0], img2.shape[1]
    img1_warped = cv.warpPerspective(img1, H, (w, h))

    imgs = np.hstack((img1_warped, img1))
    
    comparison = cv.addWeighted(img1_warped, 0.5, img2, 0.5, 0)

    img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
+ h, w를 img2의 높이와 너비로 설정하여 변환된 이미지의 크기를 img2와 동일하게 설정
+ cv.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
  + src : 변환할 원본 이미지
  + M : 3x3 변환 행렬 -> img1에 H 행렬을 적용하여 img2의 시점으로 변환
  + dsize : 변환 후 출력될 이미지 크기(width, height)
  + dst : 출력 결과를 저장할 변수
  + flags : 보간법 지정(cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4 중 선택. default=cv.INTER_LINEAR)
  + borderMode : 테두리 처리 방식 (cv.BORDER_CONSTANT, cv.BORDER_REPLICATE, cv.BORDER_REFLECT, cv.BORDER_WRAP 중 선택. default=cv.BORDER__CONSTANT)
  + borderValue : 테두리 픽셀의 색상 값(default=(0,0,0))
+ np.hstack()을 사용하여 변환된 이미지와 원본 이미지를 나란히 배치해 비교 
+ cv.addWeighted()
  + 변환된 이미지와 대상 이미지를 50%씩 혼합하여 두 이미지의 정렬 결과 확인
  + 필수 parameters : src1(첫 번째 입력 이미지), alpha(첫 번째 이미지의 가중치), src2(두 번째 입력 이미지), beta(두 번째 이미지의 가중치), gamma(밝기 조정을 위해 추가적으로 더할 값)
+ cv.drawMatches()
  + 매칭 결과 시각화

## 실행 결과
+ 변환된 이미지와 원본 이미지 비교
   <img src="https://github.com/user-attachments/assets/f0f285f7-8cec-437e-ab6d-e605cb40d6fa"/>
+ 변환된 이미지와 대상 이미지를 50%씩 혼합하여 두 이미지의 정렬 결과 확인
   <img src="https://github.com/user-attachments/assets/1e7ad0fc-89d1-4ff2-9c79-382fab8d4f71"/>
+ 매칭 결과 시각화
   <img src="https://github.com/user-attachments/assets/2ea81cc7-f80a-4a3d-a3dd-a5a5f7c53b7f"/>
