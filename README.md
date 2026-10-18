# Sign Language CV: ASL fingerspelling practice

A webcam app that recognizes American Sign Language fingerspelling letters
(A to Z, with J and Z recognized from movement) and a practice game that
asks you to make a letter and says "Correct!" when you do.

This is a **fingerspelling practice tool**, not an app that teaches ASL.

## How it works

1. **OpenCV** reads the webcam.
2. **MediaPipe** finds 21 points on your hand (`models/hand_landmarker.task`).
3. `signlingo/features.py` turns those points into 42 numbers that don't depend on
   where your hand is or how big it looks.
4. The app compares them with saved examples in `data/` and the 5 closest
   examples vote on the letter (k-nearest neighbours, no training step).
5. J and Z are recognized from the path your finger draws
   (`signlingo/motion_recognition.py`).

## Setup

Python 3.10 to 3.12.

```
pip install -r requirements.txt
```

## Run

Run these from the project folder.

| Command | What it does |
|---|---|
| `python main.py` | **SignLingo**: start menu. `1` Single Letter Practice, `2` Word Challenge, ESC quits. If you finish or leave a mode it comes back to this menu. |
| `python -m signlingo.practice` | Letter practice on its own: make the letter shown (a picture in the bottom-right shows how), hold it, get "Correct!". SPACE skips, ESC quits. |
| `python -m signlingo.practice B` | Focus mode: practice only B (or press a letter key while running; `0` goes back to random). |
| `python -m signlingo.words` | Word Challenge on its own: spell random words up to 5 letters. |
| `python -m tools.collect_data yourname_left` | Record examples one key press at a time into `data/examples_yourname_left.json`. `1` undoes, ESC quits. |
| `python -m tools.examples_tool count` | How many examples each letter has. |
| `python -m tools.make_sign_images chart.png` | Cut an ASL alphabet chart into the per-letter pictures in `assets/signs/`. |
| `python -m unittest discover tests` | Tests for J/Z motion (no webcam needed). |
| `./packaging/build_app.sh` | Build the Mac app (`dist/SignLingo.app`) and `SignLingo-mac.zip` to share. |

## Files

```
main.py                   SignLingo start menu (the app)
signlingo/                the app's code
  practice.py             Single Letter Practice
  words.py                Word Challenge
  features.py             hand -> 42 numbers, and the k-NN letter guesser
  motion_recognition.py   J and Z from finger movement
  menu_button.py          the clickable "Main Menu" button both modes use
  ui.py                   the full-screen window and the text style every screen shares
  camera.py               the webcam, opened once and shared by the menu and modes
  paths.py                where the model and examples live
models/
  hand_landmarker.task    MediaPipe hand model
assets/signs/             how to sign each letter, A.png ... Z.png (shown in letter practice)
data/
  examples_karen.json     560 webcam examples, 24 still letters
  examples_dataset.json   726 examples made from the Kaggle ASL dataset
tools/                    record and manage examples (not part of the app)
  collect_data.py         recorder, one file per person/hand
  examples_tool.py        count or remove saved examples
  make_sign_images.py     cut an alphabet chart into assets/signs/
tests/                    tests for the motion code
packaging/                SignLingo.spec and build_app.sh for the Mac app
docs/plan.md              our mentor's step-by-step guide
```

## Credits

- **Karen** (karenhuang314): the first live classifier: the 42-number
  weighted recipe and k-NN voting (now in `features.py`); recorded the
  webcam examples (`data/examples_karen.json`); the start menu and
  Word Challenge.
- **Etinosa** (EOgbeide1): J and Z motion recognition and its tests,
  the motion version of the camera app, the hand model file, and the Kaggle
  examples (`data/examples_dataset.json`).
- **Jeffrey** (JeffreySong08): first camera hand tracking, the
  practice game, the recorder and examples tool, and organizing the repo.
- Our mentor: Brandon Shin, the step-by-step guide in `docs/plan.md`.
- Hand photos for `data/examples_dataset.json` come from an ASL hand-sign
  dataset on Kaggle (add the dataset link and its license here).
- The hand pictures in `assets/signs/` are cut from a "Hand Drawn Sign Language
  Alphabet" chart found on Pinterest (add the original artist and its license here).
- Built with help from Claude (Anthropic).

Full history of who wrote what is in `git log` (use `git log --follow <file>`
for files that were moved).
