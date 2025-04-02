# 01. 소벨 에지 검출 및 결과 시각화

## 과제 설명 및 요구사항
  + 설명
     + 이미지를 그레이스케일로 변환
     + 소벨(Sobel) 필터를 사용하여 X축과 Y축 방향의 에지 검출
     + 검출된 에지 강도(edge strength) 이미지 시각화
   
  + 요구사항
      + cv.imread()를 사용하여 이미지 로드
      + cv.cvtColor()를 사용하여 그레이스케일 변환
      + cv.Sobel()을 사용하여 X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지 검출
      + cv.magnitude()를 사용하여 에지 강도 계산
      + matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
        
## 전체 코드 
   ```
     import cv2 as cv
     import sys
     import matplotlib.pyplot as plt

     img = cv.imread('./edgeDetectionImage.jpg')

     if  img is None:
         sys.exit('파일을 찾을 수 없습니다.')

     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

     grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
     grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
     
     mag = cv.magnitude(grad_x, grad_y)

     edge_strength = cv.convertScaleAbs(mag) 

     plt.subplot(1, 2, 1)
     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
     plt.title('Original')
     plt.axis('off')

     plt.subplot(1, 2, 2)
     plt.imshow(edge_strength, cmap='gray')
     plt.title('Edge Strength')
     plt.axis('off')

     plt.show()
 ```

## 이미지 로드 및 그레이스케일 변환
 ```
     img = cv.imread('./edgeDetectionImage.jpg')

     if  img is None:
         sys.exit('파일을 찾을 수 없습니다.')

     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 ```

## cv.Sobel() - 에지 검출 
 ```
     grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
     grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
 ```
+ cv.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None) -> dst 
  + <b>src</b> : 입력 영상
  + <b>ddepth</b> : 출력 영상 데이터 타입(-1이면 입력 영상과 동일한 데이터 타입 사용)
  + <b>dx</b> : X 방향 미분 차수
  + <b>dy</b> : Y 방향 미분 차수
  + <b>dst</b> : 출력 영상(행렬), 현재 코드에서는 grad_x와 grad_y에 X방향과 Y방향의 미분 결과 저장 
  + <b>ksize</b> : 커널 크기(default=3)
  + scale : 연산 결과에 추가적으로 곱할 값(default=1)
  + delta : 연산 결과에 추가적으로 더할 값(default=0)
  + borderType : 가장자리 픽셀 확장 방식(default=cv2.BORDER_DEFAULT)
+ X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출

## cv.magnitude() - 에지 강도 계산 
 ```
     mag = cv.magnitude(grad_x, grad_y)
 ```
+ cv.magnitude(x, y, magnitude=None) -> magnitude
  + x : 2D 벡터의 X좌표 행렬
  + y : 2D 벡터의 Y좌표 행렬
  + magnitude : 2D 벡터의 크기 행렬(X와 같은 크기, 같은 타입)
+ Sobel 필터로 구한 X방향, Y방향 미분 값을 cv.magnitude에 입력값으로 설정해 에지 강도 계산 

## cv.convertScaleAbs() - 에지 강도 이미지를 uint8로 변환 
 ```
     edge_strength = cv.convertScaleAbs(mag) 
 ```
+ cv.convertScaleAbs(src, dst=None, alpha=None, beta=None) -> dst
  + src : 입력 이미지
  + dst : 출력 이미지 
  + alpha : scale factor(default=1)
  + beta : 추가적으로 더할 값(default=0) 
+ 절대값을 취해 양수 영상으로 변환

## matplotlib을 사용한 결과 시각화
 ```
     plt.subplot(1, 2, 1)
     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
     plt.title('Original')
     plt.axis('off')

     plt.subplot(1, 2, 2)
     plt.imshow(edge_strength, cmap='gray')
     plt.title('Edge Strength')
     plt.axis('off')

     plt.show()
 ```
+ 원본 이미지는 cv.COLOR_BGR2RGB를 사용해 원본 컬러로 출력

## 실행 결과
   <img src="https://github.com/user-attachments/assets/5531ec81-44c9-42e9-8853-60a6073e2669"/>


# 02. 캐니 에지 및 허프 변환을 이용한 직선 검출

## 과제 설명 및 요구사항
  + 설명
     + 캐니(Canny) 에지 검출을 사용하여 에지 맵 생성
     + 허프변환(Hough Transform)을 사용하여 이미지에서 직선 검출
     + 검출된 직선을 원본 이미지에 빨간색으로 표시
   
  + 요구사항
      + cv.Canny()를 사용하여 에지 맵 생성
      + cv.HoughLinesP()를 사용하여 직선 검출
      + cv.line()을 사용하여 검출된 직선을 원본 이미지에 표시
      + matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화 
        
## 전체 코드 
   ```
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
 ```

## 이미지 로드 및 그레이스케일 변환
 ```
     img = cv.imread('./edgeDetectionImage.jpg')

     if  img is None:
         sys.exit('파일을 찾을 수 없습니다.')

     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 ```

## cv.Canny() - 에지 맵 생성
 ```
     canny = cv.Canny(gray, 100, 200)
 ```
+ cv.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None) -> edges
  + <b>image</b> : 입력 영상(보통 그레이스케일 영상을 사용) 
  + <b>threshold1</b> : 최소 임계값(이 값보다 낮은 gradient는 엣지로 간주되지 않음)
  + <b>threshold2</b> : 상단 임계값(이 값보다 높은 gradient는 엣지로 간주되지 않음)
  + edges : 에지 영상
  + apertureSize : Sobel연산을 위한 커널 크기(default=3)
  + L2gradient : True면 L2 norm, False면 L1 norm 사용(default=False)

## cv.HoughLinesP() - 직선 검출
 ```
     lines = cv.HoughLinesP(canny, 1, np.pi / 180., 100, minLineLength=35, maxLineGap=5)
 ```
+ cv.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) -> lines
  + image : 입력 에지 영상
  + rho : 누적 배열에서 rho 값의 간격(픽셀 단위의 거리 해상도, 1.0=1픽셀)
  + theta : 누적 배열에서 theta 값의 간격(라디안 단위의 각도 해상도, np.pi/180=1도)
  + threshold : 누적 배열에서 직선으로 판단할 임계값으로, 직선으로 간주되기 위한 최소 교차점 개수
  + lines : 선분의 시작과 끝 좌표(x1, y1, x2, y2) 정보를 담고 있는 ndarray
  + minLineLength : 검출할 직선의 최소 길이
  + maxLineGap : 직선으로 간주할 최대 에지 점 간격
+ 확률적 허프 변환에 의한 직선 검출 
  + cv.HoughLines(허프 변환에 의한 직선 검출) : rho와 theta 값으로 직선의 parameter 정보 제공
  + cv.HoughLinesP(확률적 허프 변환에 의한 직선 검출) : 직선의 시작과 끝 정보 제공
    
## cv.line() - 원본 이미지에 직선 그리기
 ```
     drawn_img = img.copy()
     if lines is not None:
         for line in lines:
             x1, y1, x2, y2 = line[0]
             cv.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
 ```
+ cv2.line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None) -> img
  + img : 그림을 그릴 영상
  + pt1 : 직선의 시작점, (x1, y1) 
  + pt2 : 직선의 끝점, (x2, y2) 
  + color : 선 색상, (B, G, R)
  + thickness : 선 두께(픽셀 단위, default=1)
  + lineType : 선 타입(cv2.LINE_4 or cv2.LINE_8 or cv2.LINE_AA, default=cv2.LINE_8)
  + shift : 그리기 좌표 값의 축소 비율(default=0) 
+ 원본 이미지와 직선이 그려진 이미지를 한 번에 시각화 해야하므로 원본 이미지를 복사해서 사용

## matplotlib을 사용한 결과 시각화
 ```
     plt.subplot(1, 2, 1)
     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
     plt.title('Original')
     plt.axis('off')

     plt.subplot(1, 2, 2)
     plt.imshow(edge_strength, cmap='gray')
     plt.title('Edge Strength')
     plt.axis('off')

     plt.show()
 ```
+ 원본 이미지는 cv.COLOR_BGR2RGB를 사용해 원본 컬러로 출력

## 실행 결과
   <img src="https://github.com/user-attachments/assets/52464853-a710-4824-910f-ca9a9633f3df" height="250"/>

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

