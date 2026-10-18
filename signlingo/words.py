"""
words.py - ASL fingerspelling word challenge mode for SignLingo.
Spells randomized words up to 5 characters with sequential real-time tracking.
"""

import random
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from signlingo.features import hand_to_numbers, guess_letter, load_all_examples, MOTION_LETTERS
from signlingo.menu_button import MENU_BUTTON, draw_menu_button, install
from signlingo.motion_recognition import MotionRecognizer
from signlingo.paths import MODEL_PATH

FRAMES_NEEDED = 8
WORD_CELEBRATE_SECS = 1.3

examples = load_all_examples()
LETTERS = sorted({e["letter"] for e in examples})
if not LETTERS:
    raise SystemExit("No examples found. Record some with collect_data.py first.")
LETTERS = sorted(LETTERS + list(MOTION_LETTERS))

DICTIONARY = [
    # 2-letter
    "AI", "AT", "BY", "GO", "HI", "IN", "IS", "IT", "MY", "NO", "ON", "OR", "TO", "UP",
    # 3-letter
    "ACE", "ACT", "AIR", "ARM", "BAD", "BAG", "BAR", "BAT", "BED", "BET", "BIG", "BIT",
    "BOX", "BOY", "BUS", "BUY", "CAN", "CAP", "CAR", "CAT", "CUP", "CUT", "DAY", "DOG",
    "DRY", "EAR", "EAT", "EGG", "END", "EYE", "FAN", "FAR", "FAT", "FIT", "FLY", "FOR",
    "FOX", "FUN", "GET", "HAT", "HIT", "HOT", "ICE", "INK", "JAR", "KEY", "LEG", "LIP",
    "MAP", "NET", "NUT", "PEN", "PET", "PIN", "POT", "RED", "RUN", "SIT", "SKY", "SUN",
    "TEA", "TIE", "TOY", "VAN", "WAR", "WET", "WIN", "ZIP",
    # 4-letter
    "ABLE", "ACID", "AGED", "AWAY", "BABY", "BACK", "BAKE", "BALL", "BAND", "BANK",
    "BARN", "BATH", "BEAR", "BEAT", "BIRD", "BITE", "BLUE", "BOAT", "BODY", "BONE",
    "BOOK", "BORN", "BURN", "BUSY", "BYTE", "CAFE", "CAKE", "CALL", "CALM", "CAMP",
    "CARD", "CARE", "CHAT", "CHEF", "CITY", "CLAY", "CLIP", "CLUB", "COAT", "CODE",
    "COLD", "COOK", "COOL", "COPY", "CORN", "COST", "DARK", "DATA", "DATE", "DAWN",
    "DEAL", "DEAR", "DESK", "DIET", "DISH", "DOOR", "DRAW", "DROP", "DRUM", "DUCK",
    "DUST", "DUTY", "EASY", "EDGE", "EVEN", "EXIT", "FACE", "FACT", "FAIL", "FAIR",
    "FARM", "FAST", "FEEL", "FILL", "FILM", "FIND", "FIRE", "FISH", "FIVE", "FLAG",
    "FLOW", "FOOD", "FORK", "FREE", "FROG", "FULL", "GAME", "GATE", "GIFT", "GLOW",
    "GOLD", "GOOD", "GRAB", "GRID", "GROW", "HAIR", "HALF", "HAND", "HARD", "HEAD",
    "HELP", "HERO", "HIDE", "HILL", "HOLD", "HOME", "HOPE", "HORN", "HOUR", "HUNT",
    "ICON", "IDEA", "IRON", "ITEM", "JOIN", "JUMP", "JUST", "KEEP", "KING", "KITE",
    "KNOT", "LAKE", "LAMP", "LAND", "LANE", "LEAF", "LEAN", "LION", "LOCK", "LOOP",
    "LUCK", "MAKE", "MARK", "MATH", "MEAL", "MIND", "MOON", "NAME", "NEAR", "NEST",
    "NODE", "NOTE", "OPEN", "PACK", "PAGE", "PARK", "PATH", "PLAN", "PLAY", "PLUG",
    "POST", "PUMP", "RAIN", "READ", "RING", "ROAD", "ROBOT", "ROCK", "ROOF", "ROOM",
    "ROSE", "SAFE", "SAND", "SEED", "SHIP", "SHOP", "SIGN", "SNOW", "STAR", "STEP",
    "TEAM", "TECH", "TEST", "TIME", "TREE", "UNIT", "USER", "VIEW", "WAVE", "WIND",
    # 5-letter
    "APPLE", "BEACH", "BLACK", "BRAIN", "BREAD", "BROWN", "CHAIR", "CHESS", "CLEAN",
    "CLOCK", "CLOUD", "CRAFT", "DANCE", "DRAFT", "DREAM", "DRIVE", "EARTH", "FLAME",
    "FLOAT", "FOCUS", "FRUIT", "GIANT", "GLASS", "GRAPE", "GREEN", "HEART", "HONEY",
    "HOUSE", "IMAGE", "JUICE", "LIGHT", "LOGIC", "LUNCH", "MAGIC", "MATCH", "MODEL",
    "MONEY", "MOUSE", "MUSIC", "NIGHT", "OCEAN", "PAINT", "PAPER", "PARTY", "PEACE",
    "PIANO", "PILOT", "PIZZA", "PLANT", "PLATE", "POINT", "POWER", "PRIDE", "PRIME",
    "QUICK", "QUIET", "RADIO", "RIVER", "ROBOT", "SCALE", "SCOPE", "SHARK", "SHIRT",
    "SMILE", "SMOKE", "SNAKE", "SPACE", "SPEED", "SPORT", "STAND", "STORM", "SUGAR",
    "TABLE", "TIGER", "TOUCH", "TRAIN", "TRUCK", "VALVE", "VOICE", "WATER", "WHEEL",
    "WHITE", "WOMAN", "WORLD", "YOUTH"
]

VALID_WORDS = [w for w in DICTIONARY if len(w) <= 5 and all(c in LETTERS for c in w)]


def get_random_word(previous_word=""):
    if VALID_WORDS:
        candidates = [w for w in VALID_WORDS if w != previous_word] or VALID_WORDS
        return random.choice(candidates)
    static_pool = [c for c in LETTERS if c not in MOTION_LETTERS] or LETTERS
    length = random.randint(2, min(4, len(static_pool)))
    return "".join(random.choice(static_pool) for _ in range(length))


def draw_word_hud(frame, current_word, char_idx, target, guess, streak, frames_needed, words_done):
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Glassmorphism panels
    cv2.rectangle(overlay, (20, 20), (540, 120), (24, 24, 27), -1)
    cv2.rectangle(overlay, (20, h - 85), (w - 20, h - 20), (24, 24, 27), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.rectangle(frame, (20, 20), (540, 120), (63, 63, 70), 1)
    cv2.rectangle(frame, (20, h - 85), (w - 20, h - 20), (63, 63, 70), 1)

    cv2.putText(frame, f"WORD CHALLENGE  (TARGET: {target})", (35, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (161, 161, 170), 1, cv2.LINE_AA)

    start_x = 35
    for i, ch in enumerate(current_word):
        if i < char_idx:
            color = (34, 197, 94)      # Completed: Green
        elif i == char_idx:
            color = (0, 255, 255)      # Active Target: Yellow
        else:
            color = (113, 113, 122)    # Upcoming: Dim Gray
        cv2.putText(frame, ch, (start_x + (i * 42), 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3, cv2.LINE_AA)

    status_text = f"SEE: {guess}" if guess else "SEE: ..."
    status_color = (34, 197, 94) if (guess == target and streak > 0) else (161, 161, 170)
    cv2.putText(frame, status_text, (400, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 35, 96, 470, 7
    fill_ratio = min(max(streak / frames_needed, 0.0), 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
    if fill_ratio > 0.0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * fill_ratio), bar_y + bar_h), (34, 197, 94), -1)

    footer_text = f"WORDS SPELLED: {words_done}   |   CURRENT: {current_word}"
    cv2.putText(frame, footer_text, (35, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    controls = "Press [ESC] or click MAIN MENU to go back  |  [SPACE] Skip Word"
    cv2.putText(frame, controls, (35, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (161, 161, 170), 1, cv2.LINE_AA)

    draw_menu_button(frame)
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


def run_words_mode():
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
    current_word = get_random_word()
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
                    current_word = get_random_word(previous_word=current_word)
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

        if MENU_BUTTON["clicked"]:
            MENU_BUTTON["clicked"] = False
            break
        elif key == 27:
            break
        elif key == ord(" "):
            current_word = get_random_word(previous_word=current_word)
            char_idx = 0
            streak = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_words_mode()