# Edge and Region

### 1. 소벨 에지 검출 및 결과 시각화
   
   <img src="https://github.com/user-attachments/assets/5531ec81-44c9-42e9-8853-60a6073e2669"/>

  + 이미지 로드 및 그레이스케일 변환
     ```
     img = cv.imread('./edgeDetectionImage.jpg')

     if  img is None:
       sys.exit('파일을 찾을 수 없습니다.')

     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
     ```
  + Sobel 필터를 사용해 X축과 Y축 방향의 에지 검출
     ```
     grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
     grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
     ```
  + cv.magnitude()를 사용해 에지 강도 계산
     ```
     mag = cv.magnitude(grad_x, grad_y)
     ```
  + cv.convertScaleAbs를 사용해 에지 강도 이미지를 unit8로 변환
     ```
     edge_strength = cv.convertScaleAbs(mag) 
     ```
  + 결과 시각화
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
   
### 2. 캐니 에지 및 허프 변환을 이용한 직선 검출
   
   <img src="https://github.com/user-attachments/assets/52464853-a710-4824-910f-ca9a9633f3df" height="250"/>

  + 이미지 로드 및 캐니 에지 검출을 사용한 에지 맵 생성
     ```
     img = cv.imread('./edgeDetectionImage.jpg')

     if  img is None:
         sys.exit('파일을 찾을 수 없습니다.')

     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

     canny = cv.Canny(gray, 100, 200)
     ```
  + 허프 변환을 사용한 직선 검출
     ```
     lines = cv.HoughLinesP(canny, 1, np.pi / 180., 100, minLineLength=35, maxLineGap=5)
     ```
  + cv.line()을 사용하여 원본 이미지에 직선 그리기
     ```
     drawn_img = img.copy()
     if lines is not None:
         for line in lines:
             x1, y1, x2, y2 = line[0]
             cv.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
     ```
     + 이후 원본 이미지와 직선이 그려진 이미지를 시각화 해야하므로 원본 이미지를 복사해서 사용
  + 결과 시각화
     ```
     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
     axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
     axs[0].set_title('Original Image')
     axs[0].axis('off')

     axs[1].imshow(cv.cvtColor(drawn_img, cv.COLOR_BGR2RGB))
     axs[1].set_title('Detected Lines')
     axs[1].axis('off')

     plt.show()
     ```
     
### 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출
   
   <img src="https://github.com/user-attachments/assets/493c21e4-7f8d-4304-92b2-cbb114b22baf" height="250"/>
   <img src="https://github.com/user-attachments/assets/6b37ab8b-e74b-4ae1-9f47-a649971a5d7e" height="250"/>

  + 이미지 로드 및 cv.grabCut parameter 설정
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
     + rc = (20, 20, 560, 360) 로 초기 사각형 영역 설정
  + 마스크를 사용해 원본 이미지에서 배경 제거 
     ```
     mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
     dst = src*mask2[:,:,np.newaxis]
     cv.imshow('dst', dst)
     ```
  + 마우스 콜백 함수 정의
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
  + 결과 시각화
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
     + enter를 누르면 선택한 영역에 대해 배경 제거
     + esc를 누르면 원본 이미지 마스크 이미지, 배경 제거 이미지 시각화
