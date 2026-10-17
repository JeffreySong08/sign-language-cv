"""
app.py - the website version of the project, built with Gradio.

Run:  pip install -r requirements.txt
      python app.py
Then open the link it prints (http://127.0.0.1:7860) and allow the camera.

Two tabs:
  - Live:     shows which letter you are making (like camera.py).
  - Practice: the "Make: B" -> "Correct!" game (like practice.py), with a
              letter picker for practicing one letter.

The webcam box shows the browser's own smooth video. Python only reads the
frames and sends back a small panel of text (letter, green bar, Correct!),
not a whole picture, which keeps the page fast.

It reuses the rest of the project, so nothing is duplicated:
  features.py           hand -> 42 numbers, and the k-NN letter guesser
  motion_recognition.py J and Z from finger movement (Etinosa)
  data/examples_*.json  every saved example (Karen, Kaggle, recordings)

Each visitor gets their own gr.State ("backpack") holding their motion
tracker and game score, so two people on the site never mix them up.
"""

import inspect
import random
import threading
import time
import gradio as gr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from features import hand_to_numbers, guess_letter, load_all_examples, MOTION_LETTERS
from motion_recognition import MotionRecognizer

FRAMES_NEEDED = 6       # matching frames in a row for "Correct!" (about 0.6 s)
CORRECT_SECONDS = 1.5   # how long "Correct!" stays up
STREAM_EVERY = 0.1      # seconds between webcam frames sent to Python
BOX_HEIGHT = 420        # fixed box sizes so the page doesn't jump while it updates

EXAMPLES = load_all_examples()
LETTERS = sorted({e["letter"] for e in EXAMPLES} | set(MOTION_LETTERS))
print("Letters:", " ".join(LETTERS))

# IMAGE mode treats every frame on its own, so frames from different
# visitors can't get mixed up. The lock lets only one frame use it at a time.
landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
)
landmarker_lock = threading.Lock()


def find_hand(rgb):
    """Return MediaPipe's 21 hand points for one RGB frame, or None."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())
    with landmarker_lock:
        result = landmarker.detect(image)
    return result.hand_landmarks[0] if result.hand_landmarks else None


def read_letter(frame, motion_tracker):
    """Find the hand and return (letter or None, hand or None)."""
    hand = find_hand(frame)
    h, w = frame.shape[:2]
    # The motion tracker must see every frame, hand or not, to follow a J/Z path
    motion_letter = motion_tracker.update(hand, time.monotonic(), w / h)
    if hand is None:
        return None, None
    if motion_letter:
        return motion_letter, hand
    letter, _ = guess_letter(hand_to_numbers(hand), EXAMPLES)
    return letter, hand


def panel(*rows):
    """The box next to the webcam. Its look is set in CSS below."""
    return '<div class="panel">' + "".join(rows) + "</div>"


START_PANEL = panel('<div class="hint">Press the button under the webcam to start</div>')


# ----------------- Live tab -----------------
def live_frame(frame, state):
    if frame is None:
        return state, START_PANEL
    if state is None:
        state = {"motion": MotionRecognizer()}
    letter, hand = read_letter(frame, state["motion"])
    if hand is None:
        return state, panel('<div class="hint red">Show your hand</div>')
    return state, panel('<div class="small">You are signing</div>',
                        f'<div class="big">{letter or "?"}</div>')


# ----------------- Practice tab -----------------
def new_practice_state():
    return {"motion": MotionRecognizer(), "target": random.choice(LETTERS), "streak": 0,
            "score": 0, "focus": "Random", "hits": 0, "last_took": None,
            "since": time.time(), "correct_until": 0, "skips_seen": 0}


def next_target(state):
    if state["focus"] != "Random":
        return state["focus"]
    return random.choice([L for L in LETTERS if L != state["target"]] or LETTERS)


def practice_frame(frame, focus, skips, state):
    if frame is None:
        return state, START_PANEL
    if state is None:
        state = new_practice_state()
    now = time.time()

    # The letter picker or the Skip button changed since the last frame
    if focus != state["focus"]:
        state.update(focus=focus, hits=0, last_took=None, streak=0, correct_until=0)
        state["target"] = next_target(state)
        state["since"] = now
    if skips != state["skips_seen"]:
        state["skips_seen"] = skips
        state["target"], state["streak"], state["since"] = next_target(state), 0, now

    letter, hand = read_letter(frame, state["motion"])

    if now >= state["correct_until"]:
        if state["target"] in MOTION_LETTERS:
            hit = letter == state["target"]   # one finished J/Z movement is enough
            state["streak"] = FRAMES_NEEDED if hit else 0
        else:
            state["streak"] = state["streak"] + 1 if letter == state["target"] else 0
        if state["streak"] >= FRAMES_NEEDED:
            state["score"] += 1
            state["last_took"] = now - state["since"]
            if state["focus"] != "Random":
                state["hits"] += 1
            state["streak"] = 0
            state["correct_until"] = now + CORRECT_SECONDS
            state["target"] = next_target(state)
            state["since"] = now + CORRECT_SECONDS

    score = f"Score: {state['score']}"
    if state["focus"] != "Random":
        took = f", last took {state['last_took']:.1f}s" if state["last_took"] is not None else ""
        score += f"<br>Focus {state['focus']}: {state['hits']} correct{took}"

    if now < state["correct_until"]:
        return state, panel('<div class="correct">Correct!</div>', f'<div class="small">{score}</div>')

    target = state["target"]
    tip = ""
    if target in MOTION_LETTERS:
        tip = "Draw a J with your pinky" if target == "J" else "Draw a Z with your index finger"
    seen = "Show your hand" if hand is None else f"I see: {letter or '?'}"
    bar = int(100 * state["streak"] / FRAMES_NEEDED)
    return state, panel(f'<div class="small">Make</div><div class="big">{target}</div>',
                        f'<div class="bar"><div style="width:{bar}%"></div></div>',
                        f'<div class="seen">{seen}</div>',
                        f'<div class="tip">{tip}</div>',
                        f'<div class="small">{score}</div>')


# ----------------- Page layout -----------------
# Gradio 5+ lets us choose how often frames are sent (stream_every).
# Gradio 4 (the newest one that runs on Python 3.9) doesn't have that option.
STREAM_OPTIONS = {"show_progress": "hidden"}
if "stream_every" in inspect.signature(gr.Image.stream).parameters:
    STREAM_OPTIONS["stream_every"] = STREAM_EVERY

# The panel has a fixed height so the page doesn't jump. Colors work in light and dark mode.
CSS = f"""
.panel {{ height: {BOX_HEIGHT}px; display: flex; flex-direction: column; justify-content: center;
          align-items: center; text-align: center; gap: 12px; }}
.panel .big {{ font-size: 140px; font-weight: 800; line-height: 1; }}
.panel .small {{ font-size: 20px; opacity: 0.75; }}
.panel .seen {{ font-size: 28px; font-weight: 700; color: #d4a017; }}
.panel .tip {{ font-size: 20px; color: #e8710a; min-height: 1.3em; }}
.panel .hint {{ font-size: 26px; opacity: 0.8; }}
.panel .red {{ color: #e5484d; opacity: 1; }}
.panel .correct {{ font-size: 90px; font-weight: 800; color: #22a447; }}
.panel .bar {{ width: 70%; height: 22px; border: 2px solid #888; border-radius: 11px; overflow: hidden; }}
.panel .bar div {{ height: 100%; background: #22a447; }}
"""

# Gradio 6 takes the CSS in launch(), older versions take it in Blocks()
CSS_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters

with gr.Blocks(title="ASL Fingerspelling Practice", **({} if CSS_IN_LAUNCH else {"css": CSS})) as demo:
    gr.Markdown("# ASL Fingerspelling Practice\n"
                "A practice tool for fingerspelling letters, not a way to learn ASL. "
                "Allow the camera, then press the button under the webcam to start.")

    with gr.Tab("Live"):
        with gr.Row():
            live_cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", height=BOX_HEIGHT)
            live_panel = gr.HTML(START_PANEL)
        live_state = gr.State()
        live_cam.stream(live_frame, inputs=[live_cam, live_state],
                        outputs=[live_state, live_panel], **STREAM_OPTIONS)

    with gr.Tab("Practice"):
        with gr.Row():
            focus = gr.Dropdown(["Random"] + LETTERS, value="Random", label="Letter to practice")
            skip = gr.Button("Skip letter")
        skips = gr.Number(value=0, visible=False)
        skip.click(lambda n: n + 1, skips, skips)
        with gr.Row():
            practice_cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", height=BOX_HEIGHT)
            practice_panel = gr.HTML(START_PANEL)
        practice_state = gr.State()
        practice_cam.stream(practice_frame, inputs=[practice_cam, focus, skips, practice_state],
                            outputs=[practice_state, practice_panel], **STREAM_OPTIONS)

    gr.Markdown("J and Z are recognized from movement. Built by Karen, Etinosa and Jeffrey. "
                "See README.md for credits.")

if __name__ == "__main__":
    demo.launch(**({"css": CSS} if CSS_IN_LAUNCH else {}))
