import cv2
import cvzone
import math
import time
from ultralytics import YOLO

# === YOLO + Webcam ===
cap = cv2.VideoCapture(1)
model = YOLO('yolov8n.pt')

# Load classes
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Constants
FRAME_HEIGHT = 740
FALL_Y_THRESHOLD = int(FRAME_HEIGHT * 0.85)
FALL_COOLDOWN = 30  # seconds
last_fall_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (980, FRAME_HEIGHT))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = classnames[class_id]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            aspect_ratio = height / width if width != 0 else 0

            if conf > 70 and class_name == 'person':
                # Draw box
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)

                # Posture classification
                posture = "Unknown"
                if aspect_ratio > 1.5:
                    posture = "Standing"
                elif 1.0 < aspect_ratio <= 1.5:
                    posture = "Sitting"

                # Fall detection
                is_fall = aspect_ratio < 0.7 and y2 > FALL_Y_THRESHOLD
                if is_fall:
                    posture = "⚠️ Fall Detected"
                    cvzone.putTextRect(frame, posture, [x1, y2 + 10], scale=1.5, colorR=(0, 0, 255), colorT=(255, 255, 255))

                else:
                    # Show posture if not fallen
                    cvzone.putTextRect(frame, posture, [x1, y1 - 10], scale=1.5)

    cv2.imshow('Standing, Sitting, Fall Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()