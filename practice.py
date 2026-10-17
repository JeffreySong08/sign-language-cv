"""
practice.py - fingerspelling practice mode (Step 7 of the mentor's guide).

Run:  python practice.py
  - The screen shows a letter to make, e.g. "Make: B".
  - Hold the sign steady. When the app sees that letter 10 frames in a row,
    it shows "Correct!" for 1.5 seconds and moves to the next letter.
  - J and Z are motion letters: do the whole movement (J with your pinky,
    Z with your index finger). These use Etinosa's motion_recognition.py.
  - Press n to skip a letter, ESC to quit.

It uses every data/examples_*.json file (Karen's, the Kaggle dataset, and
anything recorded with collect_data.py) and the shared recipe in features.py.
"""

import random
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import hand_to_numbers, guess_letter, load_all_examples, MOTION_LETTERS
from motion_recognition import MotionRecognizer

FRAMES_NEEDED = 10      # how many frames in a row must match
CORRECT_SECONDS = 1.5   # how long "Correct!" stays on screen

examples = load_all_examples()
LETTERS = sorted({e["letter"] for e in examples})
if not LETTERS:
    raise SystemExit("No examples found. Record some with collect_data.py first.")
LETTERS = sorted(LETTERS + list(MOTION_LETTERS))
print("Practicing letters:", " ".join(LETTERS))


def next_target(current=None):
    """Pick a random letter, but not the same one twice in a row."""
    choices = [L for L in LETTERS if L != current] or LETTERS
    return random.choice(choices)


landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
)

motion_tracker = MotionRecognizer()

cap = cv2.VideoCapture(0)
start = time.time()
last_timestamp_ms = 0

target = next_target()
streak = 0              # matching frames in a row
correct_until = 0       # time when the "Correct!" message ends
score = 0

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
    h, w = frame.shape[:2]
    now = time.time()

    guess = None
    hand = result.hand_landmarks[0] if result.hand_landmarks else None
    if hand:
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
        guess, _ = guess_letter(hand_to_numbers(hand), examples)
    # The motion tracker watches every frame for a J or Z movement
    motion_letter = motion_tracker.update(hand, time.monotonic(), w / h)
    if motion_letter:
        guess = motion_letter

    if now < correct_until:
        # Showing "Correct!" - wait, then the next letter is already set
        cv2.putText(frame, "Correct!", (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
    else:
        if target in MOTION_LETTERS:
            # One finished movement is enough for J and Z
            streak = FRAMES_NEEDED if motion_letter == target else 0
        else:
            streak = streak + 1 if guess == target else 0
        if streak >= FRAMES_NEEDED:
            score += 1
            streak = 0
            correct_until = now + CORRECT_SECONDS
            target = next_target(target)

        cv2.putText(frame, f"Make: {target}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        # Progress bar: fills up as you hold the right sign
        bar = int(300 * streak / FRAMES_NEEDED)
        cv2.rectangle(frame, (20, 95), (320, 115), (100, 100, 100), 2)
        cv2.rectangle(frame, (20, 95), (20 + bar, 115), (0, 255, 0), -1)
        seen = guess if guess else "?"
        cv2.putText(frame, f"I see: {seen}", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if target in MOTION_LETTERS:
            tip = "draw a J with your pinky" if target == "J" else "draw a Z with your index finger"
            cv2.putText(frame, f"Move: {tip}", (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(frame, f"Score: {score}   n = skip   ESC = quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Fingerspelling Practice", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:                 # ESC quits
        break
    if key == ord("n"):           # skip this letter
        target = next_target(target)
        streak = 0

cap.release()
cv2.destroyAllWindows()
