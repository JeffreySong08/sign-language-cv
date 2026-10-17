"""
collect_data.py - record examples of each letter (Step 5 of the mentor's guide).

Run:  python collect_data.py
  - Make a letter with your hand, then press that letter's key (a, b, c, d or e).
    Each press saves ONE example to examples.json.
  - Save about 20 per letter, moving your hand a little each time.
  - Every teammate should record too, so it works for different hands.
  - Saved a wrong one? Press x (or delete) to remove the LAST example.
  - Press q to quit. Your examples stay saved, and running it again adds more.
"""

import json
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import hand_to_numbers

LETTERS = "ABCDE"
EXAMPLES_FILE = "examples.json"

# ---------- Load the examples we saved last time (if any) ----------
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE) as f:
        examples = json.load(f)
else:
    examples = []  # each one: {"letter": "A", "numbers": [63 numbers]}


def save_examples():
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f)


def count(letter):
    return sum(1 for e in examples if e["letter"] == letter)


# ---------- Same hand finder as camera.py ----------
landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
)

cap = cv2.VideoCapture(0)
start = time.time()
message = "Make a letter, press its key. x = undo, q = quit"

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))
    hand = result.hand_landmarks[0] if result.hand_landmarks else None

    if hand:
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    # Show how many examples we have for each letter, e.g. "A:12  B:20  C:0 ..."
    counts = "  ".join(f"{L}:{count(L)}" for L in LETTERS)
    cv2.putText(frame, counts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):                      # q quits
        break
    if key in (ord("x"), 8, 127):            # x or delete = undo last save
        if examples:
            removed = examples.pop()
            save_examples()
            message = f"Removed last {removed['letter']} ({len(examples)} total)"
        else:
            message = "Nothing to remove"
        print(message)
    if key != 255 and chr(key).upper() in LETTERS:
        letter = chr(key).upper()
        if hand is None:
            message = "No hand found - try again"
        else:
            examples.append({"letter": letter, "numbers": hand_to_numbers(hand)})
            save_examples()
            message = f"Saved {letter} ({count(letter)} for {letter}, {len(examples)} total)"
            print(message)

cap.release()
cv2.destroyAllWindows()
