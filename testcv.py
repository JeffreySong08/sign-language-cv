import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize HandLandmarker
hand_finder = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
)

def draw_dots(frame, hand):
    h, w, _ = frame.shape
    for dot in hand:
        cx, cy = int(dot.x * w), int(dot.y * h)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_finder.detect(mp_image)

    if result.hand_landmarks:
        draw_dots(frame, result.hand_landmarks[0])

    cv2.imshow("Hand Tracking (Press Q to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()