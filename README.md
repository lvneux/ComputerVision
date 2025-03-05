# OpenCV

+ 이미지 불러오기 및 그레이스케일 변환

  + cv.imread를 사용해 이미지 로드
 
  + cv.cvtColor를 사용해 이미지를 흑백으로 변환
  
  + RGB(3차원) 이미지와 GRAY(2차원) 이미지의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 이미지를 다시 RGB로 변환
 
  + np.hstack을 사용해 이미지 연결

+ 웹캠 영상에서 에지 검출

  + cv.VideoCapture를 사용해 웹캠 영상 로드
 
  + cv.cvtColor를 사용해 웹캠 영상을 흑백으로 변환
  
  + RGB(3차원) 영상과 GRAY(2차원) 영상의 차원을 맞추기 위해 COLOR_GRAY2BGR를 사용하여 흑백 영상을 다시 RGB로 변환
 
  + cv.Canny를 사용해 에지 검출
 
  + np.hstack을 사용해 이미지 연결

+ 마우스로 영역 선택 및 ROI(관심영역) 추출

  + cv.imread를 사용해 이미지 로드
 
  + 마우스의 움직임을 따라 사각형을 그리기 위해 시작 좌표 start_x, start_y와 종료 좌표 end_x, end_y, 그리는 상태 제어를 위한 drawing 변수 사용
 
  + cv.EVENT_LBUTTONDOWN(마우스가 눌림) 상태일 때 시작 좌표를 저장하고 drawing 상태를 True로 변경
 
  + EVENT_MOUSEMOVE, drawing이 True일 때 cv.rectangle, cv.imshow를 사용해 사각형 그리기
