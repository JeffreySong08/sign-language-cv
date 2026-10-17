"""
collect_data.py - record more examples for any still letter.

Run:  python collect_data.py jeffrey_left
  - The name after the script picks the file: data/examples_jeffrey_left.json.
    Use a new name for each person / hand, e.g. karen_left, jeffrey_right.
  - Make a letter, then press its key. Each press saves ONE example.
    Holding the key down saves several in a row.
  - Move your hand a little between presses: turn it, tilt it, move it
    closer and further away. Variety is what makes recognition better.
  - Saved a wrong one? Press 1 (or delete) to remove the LAST example.
  - Press ESC to quit. Examples stay saved; running again adds more.

practice.py and camera.py automatically use every data/examples_*.json file.
J and Z are not here: they are motion letters (see motion_recognition.py).
"""

import json
import os
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import hand_to_numbers, DATA_DIR

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"   # every letter except J and Z
name = sys.argv[1] if len(sys.argv) > 1 else "new"
EXAMPLES_FILE = os.path.join(DATA_DIR, f"examples_{name}.json")

# ---------- Load the examples we saved last time (if any) ----------
if os.path.exists(EXAMPLES_FILE):
    with open(EXAMPLES_FILE) as f:
        examples = json.load(f)
else:
    examples = []  # each one: {"letter": "A", "numbers": [42 numbers]}
print(f"Saving to {EXAMPLES_FILE} ({len(examples)} examples so far)")


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
last_timestamp_ms = 0
message = "Make a letter, press its key. 1 = undo, ESC = quit"

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = max(int((time.time() - start) * 1000), last_timestamp_ms + 1)
    last_timestamp_ms = timestamp_ms
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hand = result.hand_landmarks[0] if result.hand_landmarks else None

    if hand:
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    # Counts per letter, on two lines so they fit, e.g. "A:12 B:20 C:0 ..."
    row1 = " ".join(f"{L}:{count(L)}" for L in LETTERS[:12])
    row2 = " ".join(f"{L}:{count(L)}" for L in LETTERS[12:])
    cv2.putText(frame, row1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, row2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:                            # ESC quits
        break
    if key in (ord("1"), 8, 127):            # 1 or delete = undo last save
        if examples:
            removed = examples.pop()
            save_examples()
            message = f"Removed last {removed['letter']} ({len(examples)} total)"
        else:
            message = "Nothing to remove"
        print(message)
    elif key != 255 and chr(key).upper() in LETTERS:
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
