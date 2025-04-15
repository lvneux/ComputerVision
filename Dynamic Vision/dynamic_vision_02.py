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
