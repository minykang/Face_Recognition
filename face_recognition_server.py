import os
import cv2
import numpy as np
from deepface import DeepFace
import pickle

# TensorFlow oneDNN 비활성화
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 모델 로드
model_name = 'ArcFace'

# Load the trained SVM model
model_path = "C:/svm_face_recognition_model.pickle"
with open(model_path, "rb") as f:
    clf, pca, known_encodings, known_names = pickle.load(f)

def recognize_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 얼굴 인식 및 임베딩
    results = DeepFace.represent(rgb, model_name=model_name, detector_backend='opencv', enforce_detection=False)
    
    names = []
    boxes = []
    for result in results:
        encoding = result['embedding']
        face = result['facial_area']
        boxes.append((face['y'], face['x'] + face['w'], face['y'] + face['h'], face['x'])) # (top, right, bottom, left)
        encoding_pca = pca.transform([encoding])
        probs = clf.predict_proba(encoding_pca)[0]
        name = clf.classes_[np.argmax(probs)]
        names.append(name)
    
    return names, boxes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    names, boxes = recognize_face(frame)
    
    match_found = False
    for name in names:
        if name in known_names:
            match_found = True
            break

    # 얼굴 인식된 위치에 네모난 박스 그리기
    for (top, right, bottom, left), name in zip(boxes, names):
        if match_found:
            color = (0, 255, 0)  # 녹색
            label = name
        else:
            color = (0, 0, 255)  # 빨간색
            label = "Unrecognized User"
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
