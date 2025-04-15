# 01. SORT 알고리즘을 활용한 다중 객체 추적기 구현

## 과제 설명 및 요구사항
  + 설명
     + SORT 알고리즘을 사용하여 비디오에서 다중객체를 실시간으로 추적하는 프로그램 구현 

  + 요구사항
      + 객체 검출기 구현 : YOLOv4와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체 검출
      + mathworks.comSORT 추적기 초기화 : 검출된 객체의 경계상자를 입력으로 받아 SORT 추적기 초기화
      + 객체 추적 : 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적 유지
      + 결과 시각화 : 추적된 각 객체에 고유ID를 부여하고, 해당 ID와 경계상자를 비디오 프레임에 표시하여 실시간으로 출력
        
## 전체 코드 
   ```
import numpy as np
import cv2 as cv
import sys
from sort.sort import Sort

def construct_yolo_v4():
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    
    model = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB = True)
    
    yolo_model.setInput(test_img)
    outputs = yolo_model.forward(out_layers)
    
    box,conf,id = [],[],[]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)
    
    indices = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()
    
    results = [box[i] + [conf[i]] + [id[i]] for i in indices]
    return results

model, out_layers, class_names = construct_yolo_v4()

colors = np.random.uniform(0, 255, size=(100, 3))

sort = Sort()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득 실패')

    detections = yolo_detect(frame, model, out_layers)

    persons = []
    for det in detections:
        if det[5] == 0: 
            x1, y1, x2, y2 = det[:4]
            persons.append([x1, y1, x2, y2, det[4]])  
            
    if len(persons) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(persons))

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        color = colors[int(track_id) % len(colors)]
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv.imshow('Person Tracking by SORT', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
 ```

## YOLOv4 모델 로드
 ```
def construct_yolo_v4():
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]
    
    model = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

 ```
+ coco_names.txt: COCO 데이터셋의 클래스 이름 목록 (person, car, dog 등)
+ readNet(...): YOLOv4 모델의 weight와 cfg 파일 로드
+ getUnconnectedOutLayers(): YOLO의 출력 레이어 탐색
  
## YOLO를 사용한 객체 탐지 
 ```
def yolo_detect(img, yolo_model, out_layers):
    ...
    return results
 ```
+ 입력 : 영상 프레임 img
+ 출력 : 감지된 객체의 [x1, y1, x2, y2, confidence, class_id] 리스트
+ cv.dnn.blobFromImage()를 사용해 이미지를 YOLO 입력 형식에 맞도록 변환
+ cv.dnn.NMSBoxes()를 사용해 중복 박스를 제거하여 최종 박스만 선택

## 메인 코드
 ```
model, out_layers, class_names = construct_yolo_v4()
colors = np.random.uniform(0, 255, size=(100, 3))
sort = Sort()
 ```
+ construct_yolo_v4() : YOLO 모델 로드
+ np.random.uniform(0, 255, size=(100, 3)) : ID별 색상 무작위 생성
+ Sort() : sort 초기화

## 실시간 웹캠 영상 처리
 ```
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')
 ```
+ 웹캠 영상 캡쳐 시작
  
## 실시간 탐지 및 추적 
 ```
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득 실패')

    detections = yolo_detect(frame, model, out_layers)

    persons = []
    for det in detections:
        if det[5] == 0: 
            x1, y1, x2, y2 = det[:4]
            persons.append([x1, y1, x2, y2, det[4]])  
            
    if len(persons) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(persons))

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        color = colors[int(track_id) % len(colors)]
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv.imshow('Person Tracking by SORT', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 ```
+  cap.read() : 프레임 획득
+  yolo_detect() : YOLO 탐지 수행
+  if det[5] == 0 : 'person' 클래스만 필터링
+  sort.update() : SORT 추적 수행, ID가 할당된 박스 좌표를 반환 
  
# 02. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
## 과제 설명 및 요구사항
  + 설명
     + Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 영상에 시각화하는 
프로그램 구현
   
  + 요구사항
      + Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기 초기화
      + OpenCV를 사용하여 웹캠으로부터 실시간 영상 캡처
      + 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시
      + ESC 키를 누르면 프로그램이 종료되도록 설정

## 전체 코드 
   ```
import cv2 as cv
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

left_eye = list(set(mp_face_mesh.FACEMESH_LEFT_EYE))
right_eye = list(set(mp_face_mesh.FACEMESH_RIGHT_EYE))
lips = list(set(mp_face_mesh.FACEMESH_LIPS))
nose = list(set(mp_face_mesh.FACEMESH_NOSE))

left_eye_ids = [pt for pair in left_eye for pt in pair]
right_eye_ids = [pt for pair in right_eye for pt in pair]
lips_ids = [pt for pair in lips for pt in pair]
nose_ids = [pt for pair in nose for pt in pair]

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득 실패')
        break

    result = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame,landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

            
            ih, iw, _ = frame.shape
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * iw), int(lm.y * ih)

                if idx in left_eye_ids:
                    color = (255, 0, 0)  
                elif idx in right_eye_ids:
                    color = (0, 255, 255)  
                elif idx in lips_ids:
                    color = (0, 0, 255)  
                elif idx in nose_ids:
                    color = (0, 255, 0)  
                else:
                    color = (200, 200, 200)  

                cv.circle(frame, (x, y), 1, color, -1)

    cv.imshow('Face Mesh - Colored by Landmark', cv.flip(frame, 1))

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
 ```

## MediaPipe Face Mesh 초기
 ```
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
 ```
+ mp_face_mesh : 얼굴 랜드마크를 위한 메인 기능
+ mp_drawing : 랜드마크를 프레임에 시각화할 때 사용
+ mp_styles : 랜드마크의 스타일(선/점 등)을 설정할 때 사용

## FaceMesh 객체 설
 ```
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 ```
+ max_num_faces : 동시에 감지할 얼굴 수
+ refine_landmarks=True : 눈동자, 입술 등 세밀한 랜드마크를 더 정밀하게 추적
+ min_detection_confidence : 얼굴 감지 신뢰도 임계값
+ min_tracking_confidence : 추적 신뢰도 임계값

## 눈, 코, 입 추출 - 색상 구분을 위해 랜드마크 인덱스 리스트 확보
 ```
left_eye = list(set(mp_face_mesh.FACEMESH_LEFT_EYE))
right_eye = list(set(mp_face_mesh.FACEMESH_RIGHT_EYE))
lips = list(set(mp_face_mesh.FACEMESH_LIPS))
nose = list(set(mp_face_mesh.FACEMESH_NOSE))

left_eye_ids = [pt for pair in left_eye for pt in pair]
right_eye_ids = [pt for pair in right_eye for pt in pair]
lips_ids = [pt for pair in lips for pt in pair]
nose_ids = [pt for pair in nose for pt in pair]
 ```
+ FACEMESH_LEFT_EYE, FACEMESH_LIPS 등 : 각 부위의 연결점 쌍을 정의
+ for pt in pair : 각 쌍의 점 ID를 모두 모아 단일 리스트로 변환

## 얼굴 랜드마크 추출
 ```
result = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
 ```
  
## 얼굴 랜드마크 시각
 ```
mp_drawing.draw_landmarks(image=frame,landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
 ```
+  FACEMESH_TESSELATION : 얼굴 전체에 삼각형 메시 구조를 그림
+  스타일은 mp_styles.get_default_face_mesh_tesselation_style()로 자동 지정

## 눈, 코, 입 색상 구분
 ```
for idx, lm in enumerate(face_landmarks.landmark):
    x, y = int(lm.x * iw), int(lm.y * ih)

    if idx in left_eye_ids:
        color = (255, 0, 0)  
    elif idx in right_eye_ids:
        color = (0, 255, 255)  
    elif idx in lips_ids:
        color = (0, 0, 255)  
    elif idx in nose_ids:
        color = (0, 255, 0)  
    else:
        color = (200, 200, 200)  

    cv.circle(frame, (x, y), 1, color, -1)
 ```
+ 각 랜드마크는 x, y, z 정규화 좌표값 (0~1)로 제공 : 이미지 크기로 변환
+ 왼쪽 눈 : 파란색, 오른쪽 눈 : 노란색, 입 : 빨간색, 코 : 초록색, 그외 : 회색
+ cv.circle()을 사용해 작은 원으로 시각화

## 결과 프레임 출력
```
cv.imshow('Face Mesh - Colored by Landmark', cv.flip(frame, 1))
```
+ cv.flip() : 좌우 반전으로 출력

## ESC를 누르면 프로그램 종료
```
if cv.waitKey(1) == 27:  # ESC 키
    break
cap.release()
cv.destroyAllWindows()
```
