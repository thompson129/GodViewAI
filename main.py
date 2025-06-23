import cv2
import math
from ultralytics import YOLO

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return angle

def classify_posture(angle):
    if angle < 45:
        return "Falling"
    elif angle < 70:
        return "Sitting"
    else:
        return "Standing"

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Load input video
video_path = "fall.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
falling_count = 0
fall_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xy  # [num_people, num_keypoints, 2]

        for i, kps in enumerate(keypoints):
            if len(kps) < 13:
                continue  # skip if keypoints are incomplete

            # Get body joint coordinates
            left_shoulder = kps[5]
            right_shoulder = kps[6]
            left_hip = kps[11]
            right_hip = kps[12]

            # Midpoints
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                               (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                          (left_hip[1] + right_hip[1]) / 2)

            # Angle & posture
            torso_angle = calculate_angle(hip_center, shoulder_center)
            posture = classify_posture(torso_angle)

            # Draw posture
            annotated = result.plot()
            cv2.putText(annotated, posture, (int(shoulder_center[0]), int(shoulder_center[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Fall detection
            if posture == "Falling":
                falling_count += 1
            else:
                falling_count = 0

            if 0.1 * fps <= falling_count <= 0.5 * fps:
                fall_detected = True
            if posture == "Standing":
                fall_detected = False

            if fall_detected:
                cv2.putText(annotated, "FALL DETECTED", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Show live frame
    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()