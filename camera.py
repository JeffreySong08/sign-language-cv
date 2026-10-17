"""
camera.py - live ASL fingerspelling classifier (by Karen; J/Z motion by Etinosa).

Run:  python camera.py
  - Show a letter and the window says what it sees. J and Z are recognized
    from movement (motion_recognition.py).
  - Press a letter key (not J/Z) to record 20 examples of that letter.
    New recordings go to data/examples_camera.json.
  - ESC quits.
"""

import os
import json
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from motion_recognition import MotionRecognizer
# The recipe (hand_to_numbers) and the k-NN guesser live in features.py,
# so camera.py, practice.py and collect_data.py always agree.
from features import hand_to_numbers, guess_letter, load_all_examples, DATA_DIR, FINGERTIPS

RECORD_FILE = os.path.join(DATA_DIR, "examples_camera.json")
TARGET_SAMPLES = 20

# ----------------- Step 1: Data Persistence -----------------
examples = load_all_examples()          # everything in data/, used for guessing
if os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "r") as f:
        recorded = json.load(f)         # only what this app recorded
else:
    recorded = []

def save_examples():
    with open(RECORD_FILE, "w") as f:
        json.dump(recorded, f, indent=2)

# ----------------- Step 4: MediaPipe Setup -----------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

WINDOW_NAME = "ASL Classifier"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
start = time.time()
last_timestamp_ms = 0

active_letter = None
sample_count = 0
motion_tracker = MotionRecognizer()

# ----------------- Step 5: Video Inference Loop -----------------
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    current_timestamp_ms = int((time.time() - start) * 1000)
    if current_timestamp_ms <= last_timestamp_ms:
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms

    result = landmarker.detect_for_video(mp_image, current_timestamp_ms)
    h, w = frame.shape[:2]
    hand = result.hand_landmarks[0] if result.hand_landmarks else None
    motion_letter = motion_tracker.update(
        hand if not active_letter else None, time.monotonic(), w / h
    )

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Draw bone connections
        for s_idx, e_idx in HAND_CONNECTIONS:
            p1 = (int(lm[s_idx].x * w), int(lm[s_idx].y * h))
            p2 = (int(lm[e_idx].x * w), int(lm[e_idx].y * h))
            cv2.line(frame, p1, p2, (200, 200, 200), 2)

        # Draw joints
        for idx, p in enumerate(lm):
            color = (0, 140, 255) if idx in FINGERTIPS else (0, 255, 0)
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, color, -1)

        numbers = hand_to_numbers(lm)

        # Real-time Prediction
        if not active_letter:
            predicted, conf = guess_letter(numbers, examples)
            if motion_letter:
                predicted = motion_letter
                display_text = f"Sign: {predicted} (motion)"
            elif predicted:
                display_text = f"Sign: {predicted} ({int(conf * 100)}%)"
            if predicted:
                cv2.putText(frame, display_text, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Sign: Unknown", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

        # Recording Phase
        if active_letter and sample_count < TARGET_SAMPLES:
            examples.append({"letter": active_letter, "numbers": numbers})
            recorded.append({"letter": active_letter, "numbers": numbers})
            sample_count += 1
            if sample_count >= TARGET_SAMPLES:
                save_examples()
                print(f"[STORED] 20 vectors for '{active_letter}'. Total dataset: {len(examples)}")
                active_letter = None
                sample_count = 0

    # Status Bar
    if active_letter:
        cv2.putText(frame, f"Recording '{active_letter}': {sample_count}/{TARGET_SAMPLES}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif not result.hand_landmarks:
        cv2.putText(frame, f"Samples: {len(examples)} | Press a-z to train | ESC to exit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    # ESC exits; letter keys remain available for recording.
    if key == 27:
        break
    elif ord("a") <= key <= ord("z") and active_letter is None:
        letter = chr(key).upper()
        if letter in ("J", "Z"):
            print(f"{letter} uses motion tracking, not static training. Perform the full movement.")
        else:
            active_letter = letter
            sample_count = 0

cap.release()
cv2.destroyAllWindows()