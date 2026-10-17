import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hand_numbers import hand_to_numbers
from letters import LETTERS
from examples import examples, save_examples

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
)

cap = cv2.VideoCapture(0)          # 0 = default webcam
start = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)     # mirror it
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV is BGR, MediaPipe wants RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))
    hand = result.hand_landmarks[0] if result.hand_landmarks else None
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]    
        key = cv2.waitKey(1) & 0xFF
        if hand is not None and chr(key).upper() in LETTERS:
            examples.append({"letter": chr(key).upper(), "numbers": hand_to_numbers(hand)})
            save_examples()
            print(numbers)           # 21 points, x/y from 0 to 1
        h, w = frame.shape[:2]
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

        index_up = lm[8].y < lm[6].y               # same finger rules as before
        cv2.putText(frame, f"index up: {index_up}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):          # press q to quit
        break

cap.release()
cv2.destroyAllWindows()