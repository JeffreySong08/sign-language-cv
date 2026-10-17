import os
import json
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EXAMPLES_FILE = "examples.json"
TOO_DIFFERENT = 1.5  # Rejection threshold for unrecognized hand shapes
TARGET_SAMPLES = 20

# ----------------- Step 1: Data Persistence -----------------
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE, "r") as f:
        examples = json.load(f)
else:
    examples = []

def save_examples():
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f, indent=2)

# ----------------- Step 2: Normalization -----------------
def hand_to_numbers(hand):
    wrist = hand[0]
    knuckle = hand[9]  # base of middle finger
    size = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y) or 1e-6
    numbers = []
    for dot in hand:
        numbers.append((dot.x - wrist.x) / size)
        numbers.append((dot.y - wrist.y) / size)
        numbers.append((dot.z - wrist.z) / size)
    return numbers  # 63 numbers

# ----------------- Step 3: Classification -----------------
def guess_letter(numbers):
    best_letter = None
    smallest_difference = float("inf")
    for example in examples:
        difference = 0
        for a, b in zip(numbers, example["numbers"]):
            difference += (a - b) ** 2
        if difference < smallest_difference:
            smallest_difference = difference
            best_letter = example["letter"]
    if smallest_difference > TOO_DIFFERENT:
        return None  # not close to anything
    return best_letter

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

WINDOW_NAME = "Sign Trainer"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
start = time.time()
last_timestamp_ms = 0

active_letter = None
sample_count = 0

# ----------------- Step 5: Main Loop -----------------
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

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Draw hand skeleton
        for s_idx, e_idx in HAND_CONNECTIONS:
            p1 = (int(lm[s_idx].x * w), int(lm[s_idx].y * h))
            p2 = (int(lm[e_idx].x * w), int(lm[e_idx].y * h))
            cv2.line(frame, p1, p2, (220, 220, 220), 2)

        # Draw joint landmarks
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

        numbers = hand_to_numbers(lm)

        # Live prediction when not recording
        if not active_letter:
            predicted = guess_letter(numbers)
            display_text = f"Sign: {predicted}" if predicted else "Sign: Unknown"
            cv2.putText(frame, display_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        # Record new data samples
        if active_letter and sample_count < TARGET_SAMPLES:
            examples.append({"letter": active_letter, "numbers": numbers})
            sample_count += 1
            if sample_count >= TARGET_SAMPLES:
                save_examples()
                print(f"[SAVED] 20 samples for '{active_letter}'. Total samples: {len(examples)}")
                active_letter = None
                sample_count = 0

    # UI Overlay
    if active_letter:
        cv2.putText(frame, f"Recording '{active_letter}': {sample_count}/{TARGET_SAMPLES}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif not result.hand_landmarks:
        cv2.putText(frame, f"Saved: {len(examples)} | Show hand to predict", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif ord("a") <= key <= ord("z") and active_letter is None:
        active_letter = chr(key).upper()
        sample_count = 0

cap.release()
cv2.destroyAllWindows()