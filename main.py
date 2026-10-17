"""
main.py - starts SignLingo, the downloadable app.

Right now it opens the practice game (practice.py). Later, this is where a
start menu (Learn / Practice / Challenge) will go.

Run from the code:   python main.py
Build the app:       ./build_app.sh   (makes dist/SignLingo.app on a Mac)

When the app is opened from Finder there's no terminal, so anything it
prints (including errors) goes to SignLingo-log.txt in your home folder.
"""

import os
import runpy
import subprocess
import sys
import time
import traceback

PACKAGED = hasattr(sys, "_MEIPASS")

if PACKAGED:
    log = open(os.path.join(os.path.expanduser("~"), "SignLingo-log.txt"), "w", buffering=1)
    sys.stdout = sys.stderr = log
    sys.argv = sys.argv[:1]   # macOS can add its own arguments; ignore them


def show_message(text):
    """A pop-up box on a Mac; just a printed line elsewhere."""
    print(text)
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display alert "SignLingo" message "{text}"'])


def wait_for_camera(seconds=30):
    """The first time, macOS asks for camera permission. Wait until it's allowed."""
    import cv2
    end = time.time() + seconds
    while time.time() < end:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return True
        time.sleep(1)
    return False


# In the packaged app, our files (hand_landmarker.task, data/) are unpacked
# into a special folder. Work from there so the usual file names still work.
base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
os.chdir(base)
sys.path.insert(0, base)

try:
    if not wait_for_camera():
        show_message("SignLingo can't use the camera. Allow it in System Settings, "
                     "Privacy & Security, Camera, then open SignLingo again.")
        sys.exit(1)
    runpy.run_module("practice", run_name="__main__")
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    show_message("SignLingo hit an error. Details are in SignLingo-log.txt in your home folder.")
    sys.exit(1)
