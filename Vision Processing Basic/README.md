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

   + 모폴로지 연산

      <img src="https://github.com/user-attachments/assets/e49d172b-28c3-46b9-8a3e-2de07180d986"  height="120"/>
      
     + 구조 요소를 이용해 영역의 모양을 조작
     + 영상을 변환하는 과정에서 하나의 물체가 여러 영역으로 분리되거나 한 영역으로 붙는 부작용을 없애기 위해 사용
     + 모폴로지 연산의 종류
          <img src="https://github.com/user-attachments/assets/857f3004-dff8-4310-beb4-6bb0239bf1b8"/>
         + 팽창 : 구조 요소의 중심을 1인 화소에 씌운 다음 구조 요소에 해당하는 모든 화소를 1로 바꿈
         + 침식 : 구조 요소의 중심을 1인 화소 p에 씌운 다음 구조 요소에 해당하는 모든 화소가 1인 경우에 p를 1로 유지하고 그렇지 않으면 0으로 바꿈
         + 열림 : 침식한 결과에 팽창을 적용 (원래 영역 크기 유지)
         + 닫힘 : 팽창한 결과에 침식을 적용 (원래 영역 크기 유지)
   + 영상 크롭
     ```
     b = bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
     ```
     + bin_img.shape[0] : 이미지의 세로(행) 크기
     + bin_img.shape[0]//2:bin_img.shape[0] : 이미지 세로 길이의 절반부터 끝까지(이미지의 아래쪽 절반)
     + 0:bin_img.shape[0]//2+1 : 첫 번쨰 열부터 시작해 세로 길이의 절반 +1까지(이미지의 왼쪽 절반보다 1열 더 많은 영역)
   + 5x5 커널 생성
     ```
     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
     ```
     + cv2.getStructuringElement(shape, ksizem anchor)
         + shape : 커널의 모양
             + cv2.MORPH_CROSS : 십자가형
             + cv2.MORPH_ELLIPSE : 타원형
             + cv2.MORPH_RECT : 직사각형
         + ksize : 커널의 크기
         + anchor : 커널의 기준점으로, default(-1,1)은 중심을 기준점으로 함. 이 값은 cv.MORPH_CROSS 커널을 사용할 때만 영향이 있음
   + 모폴로지 연산 -> 팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close)
     ```
     img_dilate = cv.morphologyEx(b, cv.MORPH_DILATE, kernel)
     img_erode = cv.morphologyEx(b, cv.MORPH_ERODE, kernel)
     img_open = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)
     img_close = cv.morphologyEx(b, cv.MORPH_CLOSE, kernel)
     ```
     + cv2.morphologyEx(src, op, kernel, dst, anchor, iteration, borderType, borderValue)
        + src : 입력 영상
        + op : 모폴로지 연산 종류
             + cv2.MORPH_DILATE: 팽창 연산
             + cv2.MORPH_ERODE: 침식 연산  
             + cv2.MORPH_OPEN: 열림 연산
             + cv2.MORPH_COLSE: 닫힘 연산
        + kernel : 구조화 요소 커널
        + dst, anchor, iteration, borderType, borderValue : optional. 각각 결과 영상, 커널의 기준점, 연산 반복 횟수, 외곽 영역 보정 방법, 외곽 영역 보정 값을 조정함
   + 이미지 한 줄로 배치
     ```
     imgs = np.hstack((b,img_dilate,img_erode,img_open,img_close))
     ```
     
### 3. 기하 연산 및 선형 보간 적용하기

   <img src="https://github.com/user-attachments/assets/d8891f8c-1df0-436a-a09f-149fece33ff1" height="250"/>
 
   + 기하 연산 - 회전 변환 
     ```
     rows, cols = img.shape[:2]
     rot = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)  
     ```
        + cv.getRotationMAtrix2D(center, angle, scale) : 2D 회전 변환 행렬 생성
             + center : cols/2, rows/2를 사용해 이미지의 중심을 기준으로 회전
             + angle : 45를 사용해 이미지 회전
             + scale : 1.5를 사용해 이미지의 크기를 확대
   + 선형보간 적용 
     ```
     dst = cv.warpAffine(img, rot, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)
     ```
        + cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) : 이미지를 선형 변환하는 함수
             + src : 입력 영상
             + M : Affine 변환 행렬
             + dsize : 결과 영상 크기. (w,h) 튜플 형태로 설정. (0,0)인 경우 src와 같은 크기로 설정
             + flags : 보간법 지정. default=cv2.INTER_LINEAR
             + borderMode : 가장자리 픽셀 확장 방식. default=cv2.BORDER_CONSTANT
             + borderValue : cv2.BORDER_CONSTANT일 때 사용할 상수 값. default=0(검정색)
        + cv.INTER_LINEAR를 사용해 양선형 보간법 적용
             + 양선형 보간법 : x, y 두 방향에 걸쳐 계산하는 방법으로, 화소에 걸친 비율에 따라 가중 평균하여 화솟값을 계산함 
   + 이미지를 한 화면에 출력하기 위해 img와 dst의 크기 조정(dst 이미지를 잘라 img 크기와 맞춤)
     ```
     start_x = (dst.shape[1] - cols) // 2
     start_y = (dst.shape[0] - rows) // 2
     dst_crop = dst[start_y:start_y + rows, start_x:start_x + cols]
     imgs = np.hstack((img, dst_crop))
     ```
        + 회전 후 이미지는 크기가 증가하므로, 원본 크기로 다시 잘라 중심을 맞춤
             + (start_x, start_y) : 중앙에서 원본 이미지 크기만큼 잘라낼 시작 좌표
             + start_y:start_y + rows : 세로 방향(높이)로 rows만큼 잘라냄
             + start_x:start_x + cols : 가로 방향(너비)로 cols만큼 잘라냄
      
