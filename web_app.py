import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import json
import os
import time
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "examples_karen.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in current folder.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.25
)
landmarker = vision.HandLandmarker.create_from_options(options)

examples = []
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "r") as f:
        try:
            examples = json.load(f)
        except Exception:
            examples = []
print(f"Loaded {len(examples)} gesture samples.")

history = deque(maxlen=7)
latest_features = None
current_stable_char = None
char_start_time = 0.0
assembled_sentence = ""

HOLD_DURATION = 1.0

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def extract_features(landmarks):
    """Extract scale-invariant coordinates normalized by palm size."""
    wrist = landmarks[0]
    middle_base = landmarks[9]
    # Hand scale reference (wrist to middle knuckle)
    scale = np.sqrt((middle_base.x - wrist.x)**2 + (middle_base.y - wrist.y)**2)
    if scale == 0:
        scale = 1.0

    features = []
    fingertips = {4, 8, 12, 16, 20}
    for i, pt in enumerate(landmarks):
        dx = ((pt.x - wrist.x) / scale) * (2.0 if i in fingertips else 1.0)
        dy = ((pt.y - wrist.y) / scale) * (2.0 if i in fingertips else 1.0)
        features.extend([dx, dy])
    return features

def classify(features, k=5, max_distance_threshold=8.0):
    """k-NN classification with adaptive k for small dataset counts."""
    if not examples:
        return "-", 0.0, "Dataset empty"

    feat_vec = np.array(features)
    dists = []
    for s in examples:
        dist = np.linalg.norm(feat_vec - np.array(s["numbers"]))
        dists.append((dist, s["letter"]))

    dists.sort(key=lambda x: x[0])
    
    # Adaptive k if dataset is smaller than 5
    effective_k = min(k, len(dists))
    top_k = dists[:effective_k]

    # Reject if nearest sample is too dissimilar
    if top_k[0][0] > max_distance_threshold:
        return "-", 0.0, f"No match (nearest dist: {top_k[0][0]:.2f})"

    votes = Counter([letter for _, letter in top_k])
    best, count = votes.most_common(1)[0]
    conf = count / effective_k
    debug_info = f"Top {effective_k}: " + ", ".join([f"{l} ({d:.2f})" for d, l in top_k])
    return best, conf, debug_info

def predict_stream(image):
    global latest_features, current_stable_char, char_start_time, assembled_sentence

    if image is None:
        blank = np.zeros((360, 480, 3), dtype=np.uint8)
        return blank, "-", "0%", assembled_sentence, "Waiting for video..."

    img = np.array(image, dtype=np.uint8, copy=True)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img = np.ascontiguousarray(img)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = landmarker.detect(mp_image)

    raw_letter, raw_conf, debug_str = "-", 0.0, "Searching..."

    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]

        for p1_i, p2_i in CONNECTIONS:
            p1, p2 = hand[p1_i], hand[p2_i]
            cv2.line(img, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (16, 185, 129), 3)

        for i, pt in enumerate(hand):
            color = (245, 158, 11) if i in {4, 8, 12, 16, 20} else (6, 182, 212)
            cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 6, color, -1)
            cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 7, (255, 255, 255), 1)

        latest_features = extract_features(hand)
        raw_letter, raw_conf, debug_str = classify(latest_features)
    else:
        latest_features = None

    # Temporal Smoothing
    history.append(raw_letter)
    counts = Counter(history)
    smooth_letter, most_freq = counts.most_common(1)[0]
    smooth_conf = (most_freq / len(history)) if smooth_letter != "-" else 0.0

    # Sentence Builder with Hold Debounce
    now = time.time()
    if smooth_letter != "-" and smooth_conf >= 0.7:
        if smooth_letter == current_stable_char:
            if now - char_start_time >= HOLD_DURATION:
                if smooth_letter == "SPACE":
                    assembled_sentence += " "
                else:
                    assembled_sentence += smooth_letter
                char_start_time = now + 1000.0
        else:
            current_stable_char = smooth_letter
            char_start_time = now
    else:
        current_stable_char = None
        char_start_time = now

    return img, smooth_letter, f"{int(smooth_conf * 100)}%", assembled_sentence, debug_str

def record_sample(target_char):
    global latest_features, examples
    target = target_char.strip().upper()
    if not target:
        return "Enter a character label first.", get_dataset_summary()
    if latest_features is None:
        return "No hand in frame. Position hand to capture.", get_dataset_summary()

    examples.append({"letter": target, "numbers": latest_features})
    with open(DATASET_PATH, "w") as f:
        json.dump(examples, f, indent=2)

    return f"Recorded '{target}'!", get_dataset_summary()

def get_dataset_summary():
    if not examples:
        return "Dataset empty."
    counts = Counter([s["letter"] for s in examples])
    return " | ".join([f"{k}: {v}" for k, v in sorted(counts.items())])

def clear_dataset():
    global examples
    examples = []
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w") as f:
            json.dump([], f)
    return "Dataset cleared.", "Dataset empty."

with gr.Blocks(title="ASL Vision Translator") as demo:
    gr.Markdown("# ASL Real-Time Vision Translator")
    
    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(sources=["webcam"], type="numpy", label="Webcam", streaming=True)
        with gr.Column(scale=1):
            feed = gr.Image(label="Tracking Overlay")
            with gr.Row():
                pred = gr.Textbox(label="Detected Sign (Smoothed)", value="-")
                conf_box = gr.Textbox(label="Confidence", value="0%")
            knn_debug = gr.Textbox(label="Nearest Neighbors Breakdown", value="Ready")

    with gr.Row():
        sentence_box = gr.Textbox(label="Sentence Output", value="", interactive=False)
    with gr.Row():
        btn_space = gr.Button("Add Space")
        btn_backspace = gr.Button("Backspace")
        btn_clear = gr.Button("Clear Sentence", variant="stop")

    gr.Markdown("### Record New Signs (A–Z or SPACE)")
    with gr.Row():
        label_input = gr.Textbox(label="Letter to Record", placeholder="A, B, C, or SPACE", scale=2)
        btn_record = gr.Button("Capture Current Pose", variant="primary", scale=1)
        btn_reset_data = gr.Button("Reset Dataset", variant="stop", scale=1)
    with gr.Row():
        record_status = gr.Textbox(label="Record Status", value="Ready", scale=2)
        dataset_counts = gr.Textbox(label="Current Dataset Counts", value=get_dataset_summary(), scale=3)

    cam.stream(
        fn=predict_stream,
        inputs=[cam],
        outputs=[feed, pred, conf_box, sentence_box, knn_debug],
        stream_every=0.08
    )
    btn_record.click(fn=record_sample, inputs=[label_input], outputs=[record_status, dataset_counts])
    btn_reset_data.click(fn=clear_dataset, outputs=[record_status, dataset_counts])
    btn_space.click(fn=lambda: globals().update(assembled_sentence=globals()['assembled_sentence'] + " ") or globals()['assembled_sentence'], outputs=[sentence_box])
    btn_backspace.click(fn=lambda: globals().update(assembled_sentence=globals()['assembled_sentence'][:-1]) or globals()['assembled_sentence'], outputs=[sentence_box])
    btn_clear.click(fn=lambda: globals().update(assembled_sentence="") or "", outputs=[sentence_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7865)