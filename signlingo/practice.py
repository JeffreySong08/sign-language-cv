"""
practice.py - ASL fingerspelling single letter practice mode.
"""

import random
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from signlingo.features import hand_to_numbers, guess_letter, load_all_examples, MOTION_LETTERS
from signlingo.menu_button import MENU_BUTTON, draw_menu_button, install
from signlingo.motion_recognition import MotionRecognizer
from signlingo.paths import MODEL_PATH

FRAMES_NEEDED = 10
CORRECT_SECONDS = 1.3

examples = load_all_examples()
LETTERS = sorted({e["letter"] for e in examples})
if not LETTERS:
    raise SystemExit("No examples found. Record some with collect_data.py first.")
LETTERS = sorted(LETTERS + list(MOTION_LETTERS))


def draw_styled_hud(frame, target, guess, streak, frames_needed, is_motion, motion_tip, focus, focus_hits, last_took, score):
    h, w, _ = frame.shape
    overlay = frame.copy()

    panel_h = 145 if is_motion else 120
    cv2.rectangle(overlay, (20, 20), (450, panel_h), (24, 24, 27), -1)
    cv2.rectangle(overlay, (20, h - 85), (w - 20, h - 20), (24, 24, 27), -1)

    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.rectangle(frame, (20, 20), (450, panel_h), (63, 63, 70), 1)
    cv2.rectangle(frame, (20, h - 85), (w - 20, h - 20), (63, 63, 70), 1)

    cv2.putText(frame, "SIGNLINGO PRACTICE", (35, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (161, 161, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f"SIGN: {target}", (35, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    seen_text = f"SEE: {guess}" if guess else "SEE: ..."
    status_color = (34, 197, 94) if (guess == target and streak > 0) else (161, 161, 170)
    cv2.putText(frame, seen_text, (260, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 35, 96, 380, 7
    fill_ratio = min(max(streak / frames_needed, 0.0), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
    if fill_ratio > 0.0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * fill_ratio), bar_y + bar_h), (34, 197, 94), -1)

    if is_motion and motion_tip:
        cv2.putText(frame, f"Tip: {motion_tip}", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (251, 191, 36), 1, cv2.LINE_AA)

    if focus:
        took_str = f" | Last: {last_took:.1f}s" if last_took is not None else ""
        stats_line = f"FOCUS MODE ({focus}): {focus_hits} hits{took_str}"
    else:
        stats_line = f"SCORE: {score}"

    cv2.putText(frame, stats_line, (35, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    controls = "[ESC] or MAIN MENU   |   [0] Random   |   [A-Z] Focus   |   [SPACE] Skip"
    cv2.putText(frame, controls, (35, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (161, 161, 170), 1, cv2.LINE_AA)

    draw_menu_button(frame)
    return frame


def draw_success_banner(frame):
    h, w, _ = frame.shape
    overlay = frame.copy()
    box_w, box_h = 420, 110
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (20, 83, 45), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (34, 197, 94), 2)
    cv2.putText(frame, "CORRECT!", (x1 + 45, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def next_target_letter(current=None, focus=None):
    if focus:
        return focus
    choices = [L for L in LETTERS if L != current] or LETTERS
    return random.choice(choices)


def run_practice_mode():
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )
    motion_tracker = MotionRecognizer()
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    last_timestamp_ms = 0

    install("SignLingo")

    # Focus mode: python -m signlingo.practice B  (or press a letter key while running)
    focus = sys.argv[1].upper() if len(sys.argv) > 1 else None
    if focus and focus not in LETTERS:
        raise SystemExit(f"No examples for {focus}. Letters available: {' '.join(LETTERS)}")
    focus_hits = 0
    last_took = None
    target_since = time.time()
    target = next_target_letter(focus=focus)
    streak = 0
    correct_until = 0
    score = 0

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

        if now < correct_until:
            frame = draw_success_banner(frame)
        else:
            if target in MOTION_LETTERS:
                streak = FRAMES_NEEDED if motion_letter == target else 0
            else:
                streak = streak + 1 if guess == target else 0

            if streak >= FRAMES_NEEDED:
                score += 1
                streak = 0
                correct_until = now + CORRECT_SECONDS
                last_took = now - target_since
                if focus:
                    focus_hits += 1
                target = next_target_letter(target, focus=focus)
                target_since = now + CORRECT_SECONDS

            is_motion = target in MOTION_LETTERS
            tip = "Trace J with pinky" if target == "J" else ("Trace Z with index" if target == "Z" else "")

            frame = draw_styled_hud(
                frame=frame,
                target=target,
                guess=guess,
                streak=streak,
                frames_needed=FRAMES_NEEDED,
                is_motion=is_motion,
                motion_tip=tip,
                focus=focus,
                focus_hits=focus_hits,
                last_took=last_took,
                score=score
            )

        cv2.imshow("SignLingo", frame)
        key = cv2.waitKey(1) & 0xFF

        if MENU_BUTTON["clicked"]:
            MENU_BUTTON["clicked"] = False
            break
        elif key == 27:
            break
        elif key == ord(" "):
            target = next_target_letter(target, focus=focus)
            streak = 0
            target_since = time.time()
        elif key == ord("0"):
            focus = None
            target = next_target_letter(target)
            streak = 0
            target_since = time.time()
        elif key != 255 and chr(key).upper() in LETTERS:
            focus = chr(key).upper()
            focus_hits, last_took = 0, None
            target = focus
            streak = 0
            correct_until = 0
            target_since = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_practice_mode()