"""
words.py - ASL fingerspelling word challenge mode for SignLingo.
Spells out complete words sequentially with real-time feedback.
"""

import os
import random
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import hand_to_numbers, guess_letter, load_all_examples, MOTION_LETTERS
from motion_recognition import MotionRecognizer

# Bundle path resolution
if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

FRAMES_NEEDED = 8
WORD_CELEBRATE_SECS = 1.3

examples = load_all_examples()
LETTERS = sorted({e["letter"] for e in examples})
if not LETTERS:
    raise SystemExit("No examples found. Record some with collect_data.py first.")
LETTERS = sorted(LETTERS + list(MOTION_LETTERS))

# Static dictionary filtered to letters that actually exist in your dataset
PRESET_WORDS = [
    "AI", "CAT", "DOG", "BYTE", "CODE", "DATA", "ROBOT",
    "VISION", "TECH", "OPEN", "FAST", "HAND", "SIGN"
]
WORDS = [w for w in PRESET_WORDS if all(c in LETTERS for c in w)]

# Bulletproof fallback if none of the dictionary words match dataset letters
if not WORDS:
    avail = [c for c in LETTERS if c not in MOTION_LETTERS]
    if len(avail) >= 3:
        WORDS = [avail[0] + avail[1], avail[1] + avail[2], avail[0] + avail[2]]
    elif len(avail) == 2:
        WORDS = [avail[0] + avail[1]]
    else:
        WORDS = [LETTERS[0] * 2]

print("Loaded Word Challenge words:", WORDS)


def draw_word_hud(frame, current_word, char_idx, target, guess, streak, frames_needed, words_done):
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Glassmorphism panels
    cv2.rectangle(overlay, (20, 20), (520, 120), (24, 24, 27), -1)
    cv2.rectangle(overlay, (20, h - 85), (w - 20, h - 20), (24, 24, 27), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Borders
    cv2.rectangle(frame, (20, 20), (520, 120), (63, 63, 70), 1)
    cv2.rectangle(frame, (20, h - 85), (w - 20, h - 20), (63, 63, 70), 1)

    # Top Panel Header
    cv2.putText(frame, f"WORD CHALLENGE  (TARGET LETTER: {target})", (35, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (161, 161, 170), 1, cv2.LINE_AA)

    # Word Progression
    start_x = 35
    for i, ch in enumerate(current_word):
        if i < char_idx:
            color = (34, 197, 94)      # Completed: Green
        elif i == char_idx:
            color = (0, 255, 255)      # Active: Yellow
        else:
            color = (113, 113, 122)    # Remaining: Muted Gray
        cv2.putText(frame, ch, (start_x + (i * 44), 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3, cv2.LINE_AA)

    # Detection status badge
    status_text = f"SEE: {guess}" if guess else "SEE: ..."
    status_color = (34, 197, 94) if (guess == target and streak > 0) else (161, 161, 170)
    cv2.putText(frame, status_text, (370, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    # Streak Progress Bar
    bar_x, bar_y, bar_w, bar_h = 35, 96, 450, 7
    fill_ratio = min(max(streak / frames_needed, 0.0), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
    if fill_ratio > 0.0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * fill_ratio), bar_y + bar_h), (34, 197, 94), -1)

    # Bottom Footer
    footer_text = f"WORDS SPELLED: {words_done}   |   CURRENT: {current_word}"
    cv2.putText(frame, footer_text, (35, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    controls = "[SPACE] Skip Word   |   [ESC] Quit"
    cv2.putText(frame, controls, (35, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (161, 161, 170), 1, cv2.LINE_AA)

    return frame


def draw_success_card(frame, word):
    h, w, _ = frame.shape
    overlay = frame.copy()
    box_w, box_h = 440, 110
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (20, 83, 45), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (34, 197, 94), 2)
    cv2.putText(frame, f"SPELLED '{word}'!", (x1 + 35, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


# Initialize Hand Landmarker and Motion Tracker directly at top level
landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
)
motion_tracker = MotionRecognizer()
cap = cv2.VideoCapture(0)
start_time = time.time()
last_timestamp_ms = 0

word_idx = 0
current_word = WORDS[word_idx]
char_idx = 0
words_done = 0
streak = 0
celebrate_until = 0
just_spelled = ""

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = max(int((time.time() - start_time) * 1000), last_timestamp_ms + 1)
    last_timestamp_ms = timestamp_ms
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    h, w = frame.shape[:2]
    now = time.time()

    guess = None
    hand = result.hand_landmarks[0] if result.hand_landmarks else None
    if hand:
        for p in hand:
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 5, (16, 185, 129), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
        guess, _ = guess_letter(hand_to_numbers(hand), examples)

    motion_letter = motion_tracker.update(hand, time.monotonic(), w / h)
    if motion_letter:
        guess = motion_letter

    target = current_word[char_idx]

    if now < celebrate_until:
        frame = draw_success_card(frame, just_spelled)
    else:
        if target in MOTION_LETTERS:
            streak = FRAMES_NEEDED if motion_letter == target else 0
        else:
            streak = streak + 1 if guess == target else 0

        if streak >= FRAMES_NEEDED:
            streak = 0
            char_idx += 1
            if char_idx >= len(current_word):
                words_done += 1
                just_spelled = current_word
                celebrate_until = now + WORD_CELEBRATE_SECS
                word_idx = (word_idx + 1) % len(WORDS)
                current_word = WORDS[word_idx]
                char_idx = 0

        frame = draw_word_hud(
            frame=frame,
            current_word=current_word,
            char_idx=char_idx,
            target=current_word[char_idx],
            guess=guess,
            streak=streak,
            frames_needed=FRAMES_NEEDED,
            words_done=words_done
        )

    cv2.imshow("SignLingo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(" "):
        word_idx = (word_idx + 1) % len(WORDS)
        current_word = WORDS[word_idx]
        char_idx = 0
        streak = 0

cap.release()
cv2.destroyAllWindows()