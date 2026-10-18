"""Where the hand model and examples live, so everything works from any folder.

ROOT is the project folder when running from code, and the unpacked app
folder in the built Mac app (SignLingo.spec copies files to the same layout).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                              # all examples_*.json files live here
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"  # MediaPipe hand model
SIGNS_DIR = ROOT / "assets" / "signs"                 # one picture per letter, A.png ... Z.png
