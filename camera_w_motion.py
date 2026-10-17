import os
import json
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from motion_recognition import MotionRecognizer

EXAMPLES_FILE = "examples.json"
TARGET_SAMPLES = 20
KNN_K = 5
MAX_NEIGHBOR_DISTANCE = 15.0  # Squared distance cutoff for valid gestures

# Fingertip landmark indices in MediaPipe: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
FINGERTIPS = {4, 8, 12, 16, 20}

# ----------------- Step 1: Data Persistence -----------------
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE, "r") as f:
        examples = json.load(f)
else:
    examples = []

def save_examples():
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f, indent=2)

# ----------------- Step 2: Weighted 2D Normalization -----------------
def hand_to_numbers(hand):
    wrist = hand[0]
    knuckle = hand[9]  # Base knuckle of middle finger
    size = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y) or 1e-6
    numbers = []
    
    for idx, dot in enumerate(hand):
        # Weight fingertips higher so finger configurations dominate palm geometry
        weight = 2.5 if idx in FINGERTIPS else 1.0
        numbers.append(((dot.x - wrist.x) / size) * weight)
        numbers.append(((dot.y - wrist.y) / size) * weight)
        # Depth (z) is excluded to eliminate camera-inferred jitter
    return numbers  # 42 weighted normalized numbers

# ----------------- Step 3: k-NN Multi-Class Classification -----------------
def guess_letter(numbers, k=KNN_K):
    if not examples:
        return None, 0.0

    distances = []
    for item in examples:
        # J/Z must be recognized from motion, never from a single pose.
        if item["letter"] in ("J", "Z"):
            continue
        saved_numbers = item["numbers"]
        # Skip legacy 63-element vectors if switching formats
        if len(saved_numbers) != len(numbers):
            continue
        dist = sum((a - b) ** 2 for a, b in zip(numbers, saved_numbers))
        distances.append((dist, item["letter"]))

    if not distances:
        return None, 0.0

    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]

    # Reject gesture if the best match exceeds distance ceiling
    if top_k[0][0] > MAX_NEIGHBOR_DISTANCE:
        return None, 0.0

    # Count votes among k neighbors
    votes = {}
    for _, letter in top_k:
        votes[letter] = votes.get(letter, 0) + 1

    predicted_letter = max(votes, key=votes.get)
    confidence = votes[predicted_letter] / len(top_k)
    return predicted_letter, confidence

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
            predicted, conf = guess_letter(numbers)
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