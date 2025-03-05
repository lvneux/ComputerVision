# OpenCV

+ 이미지 불러오기 및 그레이스케일 변환

   ![Image](https://github.com/user-attachments/assets/f77f1017-dd15-4a5d-8c30-2c483d1826ec)

  + cv.imread를 사용해 이미지 로드
 
  + cv.cvtColor를 사용해 이미지를 흑백으로 변환
  
  + RGB(3차원) 이미지와 GRAY(2차원) 이미지의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 이미지를 다시 RGB로 변환
 
  + np.hstack을 사용해 이미지 연결

+ 웹캠 영상에서 에지 검출

   ![Image](https://github.com/user-attachments/assets/de145712-4c4c-47b3-9356-97e5dc60fd7a)

  + cv.VideoCapture를 사용해 웹캠 영상 로드
 
  + cv.cvtColor를 사용해 웹캠 영상을 흑백으로 변환
  
  + RGB(3차원) 영상과 GRAY(2차원) 영상의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 영상을 다시 RGB로 변환
 
  + cv.Canny를 사용해 에지 검출
 
  + np.hstack을 사용해 이미지 연결

+ 마우스로 영역 선택 및 ROI(관심영역) 추출

   ![Image](https://github.com/user-attachments/assets/f57b6eb5-81a0-4c77-9350-f202bc18e24d)   

  + cv.imread를 사용해 이미지 로드
 
  + 마우스의 움직임을 따라 사각형을 그리기 위해 시작 좌표 start_x, start_y와 종료 좌표 end_x, end_y, 그리는 상태 제어를 위한 drawing 변수 사용
 
  + img.copy()를 사용해 원본 이미지 복사해 사용 (리셋하는 경우를 위해 원본 이미지 유지)
 
  + cv.EVENT_LBUTTONDOWN(마우스가 눌림) 상태일 때 시작 좌표를 저장하고 drawing 상태를 True로 변경
 
  + cv.EVENT_MOUSEMOVE, drawing이 True일 때 cv.rectangle, cv.imshow를 사용해 사각형 그리기
 
  + EVENT_LBUTTONUP(마우스가 눌리지 않음) 상태일 때 종료 좌표를 저장하고 drawing 상태를 False로 변경
 
  + img[start_y:end_y, start_x:end_x]를 사용해 사각형 내부의 이미지 크롭
 
  + r키가 눌리면 이미지를 다시 불러와 리셋
 
  + s키가 눌리면 cv.imwrite을 사용해 이미지 저장
