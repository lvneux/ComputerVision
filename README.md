# Link
- [OpenCV](#opencv)
- [Vision Processing Basic](#vision-processing-basic)
- [Edge and Region](#edge-and-region)

# OpenCV

### 1. 이미지 불러오기 및 그레이스케일 변환
   
   <img src="https://github.com/user-attachments/assets/9ba8aa94-0dfe-475f-bae7-fe43adfbf82f" height="250"/>

  + 이미지 로드 및 사이즈 조정
     ```
     img = cv.imread('./soccer.jpg')
     img_2 = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)
     ```
  + cv.cvtColor를 사용해 이미지를 흑백으로 변환
     ```
     gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
     ```
  + RGB(3차원) 이미지와 GRAY(2차원) 이미지의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 이미지를 다시 RGB로 변환
     ```
     gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
     ```
  + np.hstack을 사용해 이미지 연결
     ```
     imgs=np.hstack((img_2, gray_img))
     ```
### 2. 웹캠 영상에서 에지 검출

   <img src="https://github.com/user-attachments/assets/de145712-4c4c-47b3-9356-97e5dc60fd7a" height="250"/>
   
  + cv.VideoCapture를 사용해 웹캠 영상 로드
     ```
     cap = cv.VideoCapture(0, cv.CAP_DSHOW)
     ```
  + cv.cvtColor를 사용해 웹캠 영상을 흑백으로 변환
     ```
     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     ```
  + cv.Canny를 사용해 에지 검출
     ```
     edges = cv.Canny(gray, 100, 200)
     ```
  + RGB(3차원) 영상과 GRAY(2차원) 영상의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 영상을 다시 RGB로 변환
     ```
     gray_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
     ```
  + np.hstack을 사용해 이미지 연결
     ```
     result=np.hstack((frame, gray_edges))
     ```
### 3. 마우스로 영역 선택 및 ROI(관심영역) 추출

   <img src="https://github.com/user-attachments/assets/db6b8860-ba49-4b0b-94e6-06157f48133d" height="250"/>

  + 마우스의 움직임을 따라 사각형을 그리기 위해 시작 좌표 start_x, start_y와 종료 좌표 end_x, end_y, 그리는 상태 제어를 위한 drawing 변수 사용
     ```
     start_x, start_y, end_x, end_y, crop_img = None, None, None, None, None
     drawing = False
     ```
  + img.copy()를 사용해 원본 이미지 복사 (리셋하는 경우를 위해 원본 이미지 유지)
     ```
     temp = img.copy()
     ```
  + cv.EVENT_LBUTTONDOWN(마우스가 눌림) 상태일 때 시작 좌표를 저장하고 drawing 상태를 True로 변경
     ```
     if event==cv.EVENT_LBUTTONDOWN:
        start_x,start_y = x,y
        drawing = True
     ```
  + cv.EVENT_MOUSEMOVE, drawing이 True일 때 cv.rectangle, cv.imshow를 사용해 사각형 그리기
     ```
     elif event==cv.EVENT_MOUSEMOVE and drawing:
        cv.rectangle(temp,(start_x, start_y),(x, y),(0,0,255),2)
        cv.imshow('Drawing', temp)
     ```
  + EVENT_LBUTTONUP(마우스가 눌리지 않음) 상태일 때 종료 좌표를 저장하고 drawing 상태를 False로 변경
     ```
     elif event==cv.EVENT_LBUTTONUP:
        end_x,end_y = x,y
        drawing = False
        crop_img = img[start_y:end_y, start_x:end_x]
        cv.rectangle(img,(start_x,start_y),(end_x,end_y),(0,0,255),2)
        cv.imshow('Drawing', img)
        cv.imshow('Cropped', crop_img)
     ```
  + 'q'를 누르면 종료, 'r'을 누르면 리셋, 's'를 누르면 이미지 저장
     ```
     while(True):
       if cv.waitKey(1)==ord('q'):
           cv.destroyAllWindows()
           break
       elif cv.waitKey(1)==ord('r'):
           img = cv.imread('./soccer.jpg')
           cv.imshow('Drawing', img)
       elif cv.waitKey(1)==ord('s'):
           cv.imwrite('crop_img.jpg', crop_img)
     ```
<br/><br/>

# Vision Processing Basic

### 1. 이진화 및 히스토그램 구하기
   <img src="https://github.com/user-attachments/assets/37a1790d-d8fa-4215-8da5-8f16992488bc" height="250"/>
   <img src="https://github.com/user-attachments/assets/60ebaec0-19c6-4767-a18f-2ef9cee42f6b" height="250"/>
   
   + 그레이스케일 변환
     ```
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     ```
   + 이진화 수행
     ```
     t, bin_img = cv.threshold(gray[:,:], 127, 255, cv.THRESH_BINARY)
     ```
     + 명암 영상 이진화</br>
     <img src="https://github.com/user-attachments/assets/62ac1be3-099c-4fcf-9105-28bff708b09a" height="50"/></br>
     : 임계값 T보다 큰 화소는 1, 그렇지 않은 화소는 0으로 변환</br>
     : f는 원래 명암 영상, b는 이진 영상
     + 임계값을 127로 지정했기 때문에 OTSU 알고리즘을 사용하지 않음
     + gray[:,:]를 사용해 그레이스케일 이미지를 그대로 전달
   + 히스토그램 계산
     ```
     h=cv.calcHist([bin_img],[0],None,[256],[0,256])
     h2=h=cv.calcHist([gray],[0],None,[256],[0,256])
     ```
     + h : 그레이스케일 이미지에 대한 히스토그램 계산 
     + h2 : 이진화된 이미지에 대한 히스토그램 계산
     + 그레이스케일 이미지는 단일 채널(1채널)으로, 채널 번호는 0만 존재함
   
    
### 2. 모폴로지 연산 적용하기

   <img src="https://github.com/user-attachments/assets/075a09ec-5ae3-454d-9706-fa1e1fea5682" width="460"/>
   <img src="https://github.com/user-attachments/assets/ccbaaef2-fb38-4ac3-810a-c5e1d28380ba" height="120" width="460"/>

   + 영상 크롭
     ```
     b = bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
     ```
   + 5x5 커널 생성
     ```
     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
     ```
   + 모폴로지 연산 -> 팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close)
     ```
     img_dilate = cv.morphologyEx(b, cv.MORPH_DILATE, kernel)
     img_erode = cv.morphologyEx(b, cv.MORPH_ERODE, kernel)
     img_open = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)
     img_close = cv.morphologyEx(b, cv.MORPH_CLOSE, kernel)
     ```
   + 이미지 한 줄로 배치
     ```
     imgs = np.hstack((b,img_dilate,img_erode,img_open,img_close))
     ```
     
### 3. 기하 연산 및 선형 보간 적용하기

   <img src="https://github.com/user-attachments/assets/d8891f8c-1df0-436a-a09f-149fece33ff1" height="250"/>
 
   + 이미지를 45도 회전 시키는 회전 변환 행렬 생성 - 회전 중심 : (cols/2, rows/2)
     ```
     rows, cols = img.shape[:2]
     rot = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)  
     ```
   + 이미지 회전 및 확대(45도 회전, 1.5배 확대) & 선형보간 적용(cv.INTER_LINEAR)
     ```
     dst = cv.warpAffine(img, rot, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)
     ```
   + 이미지를 한 화면에 출력하기 위해 img와 dst의 크기 조정(dst 이미지를 잘라 img 크기와 맞춤)
     ```
     start_x = (dst.shape[1] - cols) // 2
     start_y = (dst.shape[0] - rows) // 2
     dst_crop = dst[start_y:start_y + rows, start_x:start_x + cols]
     imgs = np.hstack((img, dst_crop))
     ```
          
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
