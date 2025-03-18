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

   + 그레이스케일 변환
     ```
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     ```
   + 이진화 수행
     ```
     t, bin_img = cv.threshold(gray[:,:], 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
     ```
   + 히스토그램 계산
     ```
     h=cv.calcHist([gray],[0],None,[256],[0,256])
     ``` 
    
### 2. 모폴로지 연산 적용하기

   <img src="https://github.com/user-attachments/assets/075a09ec-5ae3-454d-9706-fa1e1fea5682" width="460"/>
   <img src="https://github.com/user-attachments/assets/4f108a19-1e19-42d8-b8d5-e722ceccbb77" height="120" width="460"/>

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
     imgs = np.hstack((img_dilate,img_erode,img_open,img_close))
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
          
