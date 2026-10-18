"""
main.py - starts SignLingo, the downloadable app.

Opens a modern start menu to choose between:
  [1] Single Letter Practice (practice.py)
  [2] Word Challenge Mode   (words.py)

Run from code:    python main.py
Build the app:    ./build_app.sh   (makes dist/SignLingo.app on macOS)

When opened from Finder, all logs and errors are written to 
SignLingo-log.txt in your home folder.
"""

import os
import runpy
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np

PACKAGED = hasattr(sys, "_MEIPASS")

if PACKAGED:
    log = open(os.path.join(os.path.expanduser("~"), "SignLingo-log.txt"), "w", buffering=1)
    sys.stdout = sys.stderr = log
    sys.argv = sys.argv[:1]   # macOS can add its own arguments; ignore them


def show_message(text):
    """A pop-up box on macOS; printed output elsewhere."""
    print(text)
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display alert "SignLingo" message "{text}"'])


def wait_for_camera(seconds=30):
    """Wait until macOS camera permission is granted."""
    end = time.time() + seconds
    while time.time() < end:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return True
        time.sleep(1)
    return False


def choose_mode():
    """Renders a start menu to select between practice and words modes."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 27)

    # Title & Subtitle
    cv2.putText(canvas, "SIGNLINGO", (215, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Select a Practice Mode to Begin", (185, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (161, 161, 170), 1, cv2.LINE_AA)

    # Mode 1 Card: Single Letter Practice
    cv2.rectangle(canvas, (110, 195), (530, 265), (39, 39, 42), -1)
    cv2.rectangle(canvas, (110, 195), (530, 265), (63, 63, 70), 1)
    cv2.putText(canvas, "[1] Single Letter Practice", (135, 238), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (34, 197, 94), 2, cv2.LINE_AA)

    # Mode 2 Card: Word Spelling Challenge
    cv2.rectangle(canvas, (110, 290), (530, 360), (39, 39, 42), -1)
    cv2.rectangle(canvas, (110, 290), (530, 360), (63, 63, 70), 1)
    cv2.putText(canvas, "[2] Word Spelling Challenge", (135, 333), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)

    # Instructions
    cv2.putText(canvas, "Press [1] or [2] to start   |   [ESC] to Exit", (170, 425), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (113, 113, 122), 1, cv2.LINE_AA)

    cv2.imshow("SignLingo", canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            cv2.destroyAllWindows()
            return "practice"
        elif key == ord("2"):
            cv2.destroyAllWindows()
            return "words"
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)


# In the packaged app, files are unpacked into a temporary folder (_MEIPASS).
base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
os.chdir(base)

try:
    if not wait_for_camera():
        show_message("SignLingo can't use the camera. Allow it in System Settings, "
                     "Privacy & Security, Camera, then open SignLingo again.")
        sys.exit(1)

    selected_mode = choose_mode()
    runpy.run_module(selected_mode, run_name="__main__")
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    show_message("SignLingo hit an error. Details are in SignLingo-log.txt in your home folder.")
    sys.exit(1)