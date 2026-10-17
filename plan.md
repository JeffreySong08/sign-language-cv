# Sign Language Practice App in Python (Beginner Guide)

A Python program that opens a camera window, puts green dots on your hand, and tells you when you've made an ASL letter correctly. **One file, one language, no website.**

The finished, tested code is in [sign_tutor.py](sign_tutor.py) (about 150 lines). This guide explains how to build it step by step. Try writing each step yourself first, and peek at the file when you're stuck. Judges will ask you to explain your code.

---

## The big idea (same as the web version)

1. **Look:** OpenCV grabs a picture from the webcam.
2. **Find the hand:** MediaPipe puts **21 dots** on your hand.
3. **Compare:** your code checks which saved letter those dots look most like.
4. **Tell the user:** if it matches the target letter, show "Correct!"

## Tech stack

| Piece | What it does |
|---|---|
| **Python 3.10 to 3.12** | The language. (3.12 is the safest pick.) |
| **OpenCV** (`opencv-python`) | Opens the webcam, shows the window, draws dots and text |
| **MediaPipe** (`mediapipe`) | Google's free tool that finds the 21 hand dots |
| **Your own code** | Saves letters to `examples.json` and finds the closest match |
| **VS Code** | Where you write and run it |

---

## Setup (Friday night, every laptop)

Open a terminal in your project folder and run:

```bash
python -m venv .venv
# Mac:      source .venv/bin/activate
# Windows:  .venv\Scripts\activate
pip install mediapipe opencv-python
```

Then test it:

```bash
python -c "import mediapipe, cv2; print('ready!', mediapipe.__version__)"
```

✅ You see `ready!` and a version number.

> ⚠️ **Intel Macs (older Macs, not M1/M2/M3/M4):** the newest MediaPipe doesn't install on them. Use Python 3.11 and run `pip install mediapipe==0.10.21 opencv-python` instead. The code works the same. Check with  → About This Mac: "Chip: Apple" means you're fine.

> ⚠️ **Don't copy old tutorials that use `mp.solutions.hands`.** That was removed in MediaPipe 1.0, so that code crashes. Use the `HandLandmarker` code in this guide.

---

## Step by step

Test after every step before moving on.

### Step 1: Show the webcam

```python
import cv2

camera = cv2.VideoCapture(0)       # 0 = your built-in webcam

while True:
    ok, frame = camera.read()      # grab one picture
    if not ok:
        break
    frame = cv2.flip(frame, 1)     # mirror it, like a selfie
    cv2.imshow("Sign Tutor", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key quits
        break

camera.release()
cv2.destroyAllWindows()
```

✅ A window shows you. ESC closes it.

> **Mac:** the first time, macOS asks to let **Terminal** (or VS Code) use the camera. Click Allow. If you missed it: System Settings → Privacy & Security → Camera → turn it on, then restart the terminal.

### Step 2: Load MediaPipe

At the top of the file, add the imports and load the hand finder (the file downloads the model the first time). Copy this part from `sign_tutor.py`, lines 6 to 37.

Add `start_time = time.time()` just before `while True:`. Then, inside the loop after `cv2.flip`, ask MediaPipe for the dots:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe wants RGB colors
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
timestamp_ms = int((time.time() - start_time) * 1000)
result = hand_finder.detect_for_video(image, timestamp_ms)
hand = result.hand_landmarks[0] if result.hand_landmarks else None
print("hand!" if hand else "no hand")
```

✅ The terminal prints `hand!` when your hand is in view.

### Step 3: Draw the dots

```python
def draw_dots(frame, hand):
    height, width, _ = frame.shape
    for dot in hand:
        x = int(dot.x * width)     # MediaPipe gives 0 to 1, so multiply by the size
        y = int(dot.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
```

Call `draw_dots(frame, hand)` when `hand` isn't `None`, before `cv2.imshow`.

✅ Green dots follow your fingers. The computer vision part is done! 🎉

### Step 4: Turn the dots into numbers

Measure every dot from the wrist and divide by the hand's size, so a hand close to the camera and far from it give similar numbers. See `hand_to_numbers` in the file.

### Step 5: Save letters

When you press a letter key, save the current hand's numbers with that letter, and write them all to `examples.json`.

```python
key = cv2.waitKey(1) & 0xFF
if hand is not None and chr(key).upper() in LETTERS:
    examples.append({"letter": chr(key).upper(), "numbers": hand_to_numbers(hand)})
    save_examples()
```

Save **about 20 per letter**, moving your hand a little each time. **Every teammate records too.**

✅ The terminal prints `Saved L (20 examples total)`.

### Step 6: Guess the letter

Compare the live hand to every saved example and pick the one with the smallest difference. If nothing is close (`TOO_DIFFERENT`), return `None`. See `guess_letter` in the file.

> **Tuning tip:** `print(smallest_difference)` while making right and wrong letters, then pick a cutoff between them.

✅ The window says "Looks like L..." when you make an L.

### Step 7: Say "Correct!"

Only count it if the guess matches the target for **10 frames in a row**. Then show "Correct!" for 1.5 seconds and move to the next letter. Use `cv2.putText` to write on the window.

✅ It says "Correct!" and moves on. **This is your demo.**

---

## 12-hour plan (team of 3)

| Time | 🎥 Camera person | 🔢 Data person | 🎨 Screen + pitch person |
|---|---|---|---|
| 10–12 | Steps 1–2 | Reads steps 4–6, helps with setup problems | Picks letter pictures (Deaf-led source like Lifeprint), plans what the window shows |
| 12–2 | Step 3 | Steps 4–5 | Text and colors on screen, tries to reach an ASL signer |
| 2–4 | Records examples with everyone | Step 6 | Shows the letter picture in the corner of the window |
| 4–6 | Step 7 together | Tunes the cutoff | Connects screen ideas to the code |
| 6–8 | Bug fixes, test on all 3 hands | Re-records letters that get mixed up | Writes the Devpost |
| 8–9 | Puts the code on GitHub | Records a **screen video** of it working | Finishes the Devpost and demo script |
| 9–10 | Everyone: submit, practice the demo, rest | | |

**Checkpoints:**
- **12 PM: green dots follow your hand.** If not, get a mentor right away.
- **4 PM: guesses at least 3 letters.** If not, drop to B, L and V.
- **6 PM: "Correct!" works.** If not, just show the guessed letter live.
- **8 PM: stop adding features.**

**For judging:** there's no link to click, so bring a charged laptop to demo live, and put the screen video on Devpost.

---

## When things go wrong

| Problem | Fix |
|---|---|
| `pip install mediapipe` fails | Check your Python version (`python --version`). Use 3.10 to 3.12. Intel Mac: see the setup note. |
| `module 'mediapipe' has no attribute 'solutions'` | You copied an old tutorial. Use this guide's `HandLandmarker` code. |
| Black window or "Couldn't read from the camera" | Close Zoom/FaceTime. On Mac, allow camera for Terminal/VS Code. Try `VideoCapture(1)`. |
| Model download fails (SSL error on Mac) | Run "Install Certificates.command" in your Python folder, or download `hand_landmarker.task` from the link in the code and put it next to `sign_tutor.py`. |
| Always guesses the same letter | More examples from more people. Lower `TOO_DIFFERENT`. |
| Mixes up two letters | Those letters look alike. Remove one for now. |

---

## Be respectful about ASL

Call it a **fingerspelling practice tool**, not an app that "teaches ASL." Use Deaf-led sources for letter pictures and credit them, ask a Deaf or ASL signer to check your letters if you can, and be honest about how well it works ("tested on 5 people, 8 out of 10 right").
