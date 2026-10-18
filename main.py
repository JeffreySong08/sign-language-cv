"""
main.py - starts SignLingo, the downloadable app.

Opens a modern start menu to choose between:
  [1] Learn Letters (signlingo/practice.py)
  [2] Spell Words   (signlingo/words.py)

Run from code:    python main.py
Build the app:    ./packaging/build_app.sh   (makes dist/SignLingo.app on macOS)

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

from signlingo import camera, ui
from signlingo.settings_ui import SettingsPanel, window_is_open

PACKAGED = hasattr(sys, "_MEIPASS")

if PACKAGED:
    log = open(os.path.join(os.path.expanduser("~"), "SignLingo-log.txt"), "w", buffering=1)
    sys.stdout = sys.stderr = log
    sys.argv = sys.argv[:1]   # macOS can add its own arguments; ignore them
    # MediaPipe loads matplotlib, and the packaged app gives matplotlib a new
    # temp folder each launch, so it re-scans every font (seconds). Keep its
    # cache instead, one per app location in case the app is moved.
    import hashlib
    mpl_cache = os.path.join(os.path.expanduser("~"), "Library", "Caches", "SignLingo",
                             "matplotlib-" + hashlib.sha1(sys._MEIPASS.encode()).hexdigest()[:12])
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cache


def show_message(text):
    """A pop-up box on macOS; printed output elsewhere."""
    print(text)
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display alert "SignLingo" message "{text}"'])


def wait_for_camera(seconds=30):
    """Wait until macOS camera permission is granted. The camera stays open
    for the mode, so it starts without reopening it."""
    end = time.time() + seconds
    while time.time() < end:
        cap = camera.open_camera()
        if cap.isOpened() and cap.read()[0]:
            return True
        camera.close_camera()
        time.sleep(1)
    return False


def choose_mode():
    """Start menu, in the same window and style as the practice screens."""
    w, h = ui.screen_size()    # layout in points; the canvas itself is full resolution
    canvas = ui.blank()

    mid = h // 2               # everything sits around the middle of the screen
    ui.centered(canvas, "SignLingo", mid - 160, 3.0, ui.WHITE, 7)
    ui.centered(canvas, "ASL fingerspelling practice", mid - 100, 1.0, ui.GREY, 2)

    # The two modes: the key to press in yellow, the name in white. Each line
    # can also be clicked.
    options = []
    for y, key, name, mode in ((mid + 40, "1", "Learn Letters", "signlingo.practice"),
                               (mid + 130, "2", "Spell Words", "signlingo.words")):
        width = ui.text_width(f"{key}   {name}", 1.5, 4)
        x = (w - width) // 2
        ui.text(canvas, key, (x, y), 1.5, ui.YELLOW, 4)
        ui.text(canvas, name, (x + ui.text_width(f"{key}   ", 1.5, 4), y), 1.5, ui.WHITE, 4)
        options.append(((x - 30, y - 60, x + width + 30, y + 25), mode))

    ui.help_line(canvas, "1 or 2 = start   ESC = quit")

    settings_panel = SettingsPanel()
    clicked = {"mode": None}

    def on_mouse(event, x, y, flags, param):
        x, y = ui.to_points(x, y)        # OpenCV reports real pixels; layout is in points
        if settings_panel.on_mouse(event, x, y, flags, param):   # gear / popup first
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for (x1, y1, x2, y2), mode in options:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked["mode"] = mode

    ui.open_window()
    cv2.setMouseCallback(ui.WINDOW, on_mouse)
    ui.show(settings_panel.draw(canvas.copy()))
    cv2.waitKey(1)   # draw the menu now
    # Load the hand-tracking library while the menu is up, so pressing 1 or 2
    # starts quickly. (A key pressed meanwhile still counts.)
    import mediapipe  # noqa: F401
    from mediapipe.tasks.python import vision  # noqa: F401
    while True:
        ui.show(settings_panel.draw(canvas.copy()))
        # Check often instead of waiting for a key, to notice the window being closed
        key = cv2.waitKey(30) & 0xFF
        if not window_is_open():
            return None                   # closed with the window's close button: quit
        if settings_panel.handle_key(key):
            continue                      # the settings popup takes the keys while open
        if clicked["mode"] is not None:
            return clicked["mode"]
        if key == ord("1"):
            return "signlingo.practice"   # same window carries on into the mode
        elif key == ord("2"):
            return "signlingo.words"
        elif key == 27:
            cv2.destroyAllWindows()
            return None


def main():
    # In the packaged app, files are unpacked into a temporary folder (_MEIPASS).
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)
    try:
        if not wait_for_camera():
            show_message("SignLingo can't use the camera. Allow it in System Settings, "
                         "Privacy & Security, Camera, then open SignLingo again.")
            return 1

        # The menu keeps coming back after a mode ends, so you can pick another
        # one. A mode returns here with ESC or the Main Menu button; ESC on the
        # menu itself, or closing the window, quits SignLingo.
        while True:
            selected_mode = choose_mode()
            if selected_mode is None:
                return 0
            runpy.run_module(selected_mode, run_name="__main__")
    except SystemExit:
        raise  # a mode's window was closed: quit, don't reopen the menu
    except Exception:
        traceback.print_exc()
        show_message("SignLingo hit an error. Details are in SignLingo-log.txt in your home folder.")
        return 1
    finally:
        camera.close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
