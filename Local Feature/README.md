# 01. SIFT를 이용한 특징점 검출 및 시각

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

img1 = cv.imread('./imgs/mot_color70.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./imgs/mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수 : ', len(kp1), len(kp2))

start = time.time()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
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

# 03. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

## 과제 설명 및 요구사항
  + 설명
     + 사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체 추출
     + 객체 추출 결과를 마스크 형태로 시각화
     + 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력
   
  + 요구사항
      + cv.grabCut()를 사용하여 대화식 분할 수행
      + 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
      + 마스크를 사용하여 원 본이미지에서 배경 제거
      + matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지를 나란히 시각화
        
## 전체 코드 
   ```
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

rc = (20, 20, 560, 360)

cv.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)

mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
dst = src*mask2[:,:,np.newaxis]

cv.imshow('dst', dst)

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
        mask2=np.where((mask==cv.GC_PR_BGD) | (mask==cv.GC_BGD), 0, 1).astype('uint8')     
        dst = src*mask2[:,:,np.newaxis]
        cv.imshow('dst',dst)
    elif key == 27:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(src)
        axs[0].set_title('Original Image')
        axs[1].imshow(mask2, cmap='gray')
        axs[1].set_title('Mask')
        axs[2].imshow(dst)
        axs[2].set_title('Object Extracted')
        plt.show()
        break

cv.destroyAllWindows()
 ```

## 이미지 로드 및 cv.grabCut 초기 실행
 ```
     src = skimage.data.coffee()

     mask = np.zeros(src.shape[:2], np.uint8)
     bgdModel = np.zeros((1,65), np.float64)
     fgdModel = np.zeros((1,65), np.float64)
     iterCount = 1
     mode = cv.GC_INIT_WITH_RECT

     rc = (20, 20, 560, 360)

     cv.grabCut(src, mask, rc, bgdModel, fgdModel, iterCount, mode)
 ```
+ skimage.data.coffee() : skimage에서 제공하는 이미지 로드
+ mask
  + src의 3차원 배열 형태(높이x너비x채널)에서 이미지의 크기(높이,너비)정보를 추출
  + np.zeros(src.shape[:2], np.uint8)을 통해 이미지 크기의 배열을 0으로 채움
  + np.uint8 : 0~255 범위의 정수값 저장
  + src 이미지와 동일한 크기의 2차원 배열(마스크)로 초기화되어, grabCut에서 배경(백그라운드)과 전경(포그라운드)을 구분하기 위한 마스크로 활용됨
  + mask 배열의 각 픽셀은 cv.GC_BGD(0), cv.GC_FGD(1), cv.GC_PR_BGD(2), cv.GC_PR_FGD(3)으로 분류되며, 초기에는 모든 픽셀을 cv.GC_BGD로 설정해 완전한 배경으로 가정 
+ bgdModel, fgdModel : grabCut 내부에서 사용되는 배경 및 전경 모델
+ iterCount : 1회 반복 실행
+ mode : cv.GC_INIT_WITH_RECT으로 설정하여 초기 전경 영역을 사각형(rc)으로 설정
+ rc : 좌측 상단 (20, 20)에서 가로 560, 세로 360 영역을 전경으로 설정
+ grabCut 초기 실행 후 : mask가 0~3 값으로 업데이트
  
## 마스크를 사용해 원본 이미지에서 배경 제거 
 ```
     mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
     dst = src*mask2[:,:,np.newaxis]
     cv.imshow('dst', dst)
 ```
+ mask2
  + cv.GC_BGD(0)와 cv.GC_PR_BGD(2)를 0으로, 나머지는 1로 변환
+ dst
  + src : 컬러 영상 채널을 갖는 3차원 행렬
  + mask2 : 그레이 스케일 영상의 2차원 행렬
  + newaxis를 적용하여 같은 차원으로 만든 후 곱셈
  + 배경(mask2=0)은 검은색(0), 전경(mask2=1)은 원래 색 유지
+ cv.imshow : 전경 추출된 결과 출력력

## 마우스 콜백 함수 정의 
 ```
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
 ```
+ cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None) -> img
  + img : 그림을 그릴 영상
  + center : 원의 중심 좌표, (x, y)
  + radius : 원의 반지름
  + color : 선 색상, (B, G, R)
  + thickness : 선 두께(default=1, -1로 지정하면 내부를 채움)
  + lineType : 선 타입(cv2.LINE_4 or cv2.LINE_8 or cv2.LINE_AA)
  + shift : 그리기 좌표 값의 축소 비율(default=0) 
+ cv.EVENT_LBUTTONDOWN(좌클릭)
  ```
  cv.circle(dst, (x, y), 3, (255, 0, 0), -1)
  cv.circle(mask, (x, y), 3, cv.GC_FGD, -1) 
  ```
  + 사용자가 왼쪽 버튼을 클릭하면, 해당 좌표에 파란색 점을 그림
  + mask 배열에서도 해당 좌표를 전경(GC_FGD)으로 설정
+ cv.EVENT_RBUTTONDOWN(우클릭)
  ```
  cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
  cv.circle(mask, (x, y), 3, cv.GC_BGD, -1) 
  ```
  + 사용자가 오른쪽 버튼을 클릭하면, 해당 좌표에 빨간색 점을 그림
  + mask 배열에서도 해당 좌표를 배경(GC_BGD)으로 설정
+ cv.EVENT_MOUSEMOVE(마우스 이동)
  ```
  if flags & cv.EVENT_FLAG_LBUTTON:
    cv.circle(dst, (x, y), 3, (255, 0, 0), -1) 
    cv.circle(mask, (x, y), 3, cv.GC_FGD, -1)  
    cv.imshow('dst', dst)           
  elif flags & cv.EVENT_FLAG_RBUTTON:
    cv.circle(dst, (x, y), 3, (0, 0, 255), -1)
    cv.circle(mask, (x, y), 3, cv.GC_BGD, -1)  
    cv.imshow('dst', dst)
  ```
  + flags & cv.EVENT_FLAG_LBUTTON(왼쪽 버튼을 누른 상태에서 마우스 이동) : 파란색 선이 그려지면서 mask 값이 전경(GC_FGD)으로 설정
  + flags & cv.EVENT_FLAG_RBUTTON(오른쪽 버튼을 누른 상태에서 마우스 이동) : 빨색 선이 그려지면서 mask 값이 배경(GC_BGD)으로 설정

## 키보드 이벤트 & 결과 시각화 
 ```
     while True: 
         key=cv.waitKey()
         if key == 13:
             cv.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK)
             mask2=np.where((mask==cv.GC_PR_BGD) | (mask==cv.GC_BGD), 0, 1).astype('uint8')     
             dst = src*mask2[:,:,np.newaxis]
             cv.imshow('dst',dst)
         elif key == 27:
             fig, axs = plt.subplots(1, 3, figsize=(15, 5))
             axs[0].imshow(src)
             axs[0].set_title('Original Image')
             axs[1].imshow(mask2, cmap='gray')
             axs[1].set_title('Mask')
             axs[2].imshow(dst)
             axs[2].set_title('Object Extracted')
             plt.show()
             break
 ```
+ enter(key == 13) : 선택한 영역에 대해 배경 제거
+ esc(key == 27) : 원본 이미지 마스크 이미지, 배경 제거 이미지 시각화
  
## 실행 결과
   <img src="https://github.com/user-attachments/assets/493c21e4-7f8d-4304-92b2-cbb114b22baf" height="250"/>
   <img src="https://github.com/user-attachments/assets/6b37ab8b-e74b-4ae1-9f47-a649971a5d7e" height="250"/>


