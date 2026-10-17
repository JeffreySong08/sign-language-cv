# Sign Language CV: ASL fingerspelling practice

A webcam app that recognizes American Sign Language fingerspelling letters
(A to Z, with J and Z recognized from movement) and a practice game that
asks you to make a letter and says "Correct!" when you do.

This is a **fingerspelling practice tool**, not an app that teaches ASL.

## How it works

1. **OpenCV** reads the webcam.
2. **MediaPipe** finds 21 points on your hand (`hand_landmarker.task`).
3. `features.py` turns those points into 42 numbers that don't depend on
   where your hand is or how big it looks.
4. The app compares them with saved examples in `data/` and the 5 closest
   examples vote on the letter (k-nearest neighbours, no training step).
5. J and Z are recognized from the path your finger draws
   (`motion_recognition.py`).

## Setup

Python 3.10 to 3.12.

```
pip install -r requirements.txt
```

## Run

| Command | What it does |
|---|---|
| `python practice.py` | Practice game: make the letter shown, hold it, get "Correct!". `n` skips, ESC quits. |
| `python camera.py` | Live classifier: shows the letter it sees. Press a letter key to record 20 examples (saved to `data/examples_camera.json`). ESC quits. |
| `python collect_data.py yourname_left` | Record examples one key press at a time into `data/examples_yourname_left.json`. `1` undoes, ESC quits. |
| `python examples_tool.py count` | How many examples each letter has. |
| `python -m unittest test_motion_recognition` | Tests for J/Z motion (no webcam needed). |

## Files

| File | What it is |
|---|---|
| `camera.py` | Live classifier with recording |
| `practice.py` | Practice game (the demo) |
| `collect_data.py` | Recorder, one file per person/hand |
| `features.py` | Shared hand-to-numbers recipe and letter guesser |
| `motion_recognition.py` | J and Z from finger movement |
| `test_motion_recognition.py` | Tests for the motion code |
| `examples_tool.py` | Count or remove saved examples |
| `hand_landmarker.task` | MediaPipe hand model |
| `data/examples_karen.json` | 560 webcam examples, 24 still letters |
| `data/examples_dataset.json` | 726 examples made from the Kaggle ASL dataset |
| `docs/plan.md` | Our mentor's step-by-step guide |

## Credits

- **Karen** (karenhuang314): live classifier (`camera.py`): the
  42-number weighted recipe, k-NN voting, recording, drawing; recorded the
  webcam examples (`data/examples_karen.json`).
- **Etinosa** (EOgbeide1): J and Z motion recognition and its tests,
  the motion version of the camera app, the hand model file, and the Kaggle
  examples (`data/examples_dataset.json`).
- **Jeffrey** (JeffreySong08): first camera hand tracking, the
  practice game, the recorder and examples tool, and organizing the repo.
- Our mentor: the step-by-step guide in `docs/plan.md`.
- Hand photos for `data/examples_dataset.json` come from an ASL hand-sign
  dataset on Kaggle (add the dataset link and its license here).
- Built with help from Claude (Anthropic).

Full history of who wrote what is in `git log` (use `git log --follow <file>`
for files that were moved).
