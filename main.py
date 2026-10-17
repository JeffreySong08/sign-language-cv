"""
main.py - starts SignLingo, the downloadable app.

Right now it opens the practice game (practice.py). Later, this is where a
start menu (Learn / Practice / Challenge) will go.

Run from the code:   python main.py
Build the app:       ./build_app.sh   (makes dist/SignLingo.app on a Mac)
"""

import os
import runpy
import sys

# In the packaged app, our files (hand_landmarker.task, data/) are unpacked
# into a special folder. Work from there so the usual file names still work.
base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
os.chdir(base)
sys.path.insert(0, base)

runpy.run_module("practice", run_name="__main__")
