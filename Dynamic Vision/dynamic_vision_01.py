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