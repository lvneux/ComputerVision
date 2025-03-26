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
