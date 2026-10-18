"""Settings gear and popup shared by the menu and both modes (from PR #1),
drawn in the same style as the rest of the app."""

import math
import time

import cv2

from signlingo import ui
from signlingo.settings import get_settings


ROWS = (
    ("low_power", "Low-power mode", "Limit processing to about 12 FPS"),
    ("show_landmarks", "Show hand landmarks", "Off hides the dots, not recognition"),
    ("selfie", "Selfie view", "Mirror the camera image"),
)
GEAR_SIZE = 52     # same height as the Main Menu button


def window_is_open(name=ui.WINDOW):
    """False once the window is closed (its close button), so the next
    imshow() doesn't quietly open a new one."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def limit_frame_rate(started, low_power):
    if low_power:
        time.sleep(max(0, 1 / 12 - (time.monotonic() - started)))


class SettingsPanel:
    def __init__(self, store=None):
        self.store = store if store is not None else get_settings()
        self.is_open = False
        self.revision = 0  # Opening/closing also invalidates pending sign evidence.
        self.gear_rect = None
        self.close_rect = None
        self.row_rects = []
        self.gear_hover = False

    @staticmethod
    def _inside(rect, x, y):
        return rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def toggle_open(self):
        self.is_open = not self.is_open
        self.revision += 1

    def handle_key(self, key):
        if key == ord(","):
            self.toggle_open()
            return True
        if not self.is_open:
            return False
        if key == 27:
            self.toggle_open()
        elif key in (ord("1"), ord("2"), ord("3")):
            self.store.toggle(ROWS[key - ord("1")][0])
            self.revision += 1
        return True  # Modal: keys must not select a mode or focus behind the popup.

    def on_mouse(self, event, x, y, flags=0, param=None):
        self.gear_hover = self._inside(self.gear_rect, x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.gear_hover:
                self.toggle_open()
                return True
            if self.is_open:
                if self._inside(self.close_rect, x, y):
                    self.toggle_open()
                else:
                    for key, rect in self.row_rects:
                        if self._inside(rect, x, y):
                            self.store.toggle(key)
                            self.revision += 1
                            break
                return True
        return self.is_open or self.gear_hover

    def draw(self, frame, right=None):
        """Gear in the top-right corner (left of `right`, e.g. the Main Menu
        button), plus the popup when it's open."""
        h, w = frame.shape[:2]
        x2 = (right if right is not None else w) - 8
        self.gear_rect = (x2 - GEAR_SIZE, 8, x2, 8 + GEAR_SIZE)
        x1, y1, x2, y2 = self.gear_rect
        ui.backdrop(frame, x1, y1, x2, y2, darkness=0.75 if self.gear_hover else 0.6)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = ui.YELLOW if self.gear_hover else ui.WHITE
        for angle in range(0, 360, 45):
            a = math.radians(angle)
            cv2.line(frame, (int(cx + 10 * math.cos(a)), int(cy + 10 * math.sin(a))),
                     (int(cx + 15 * math.cos(a)), int(cy + 15 * math.sin(a))), color, 4, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 10, color, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, color, 2, cv2.LINE_AA)
        if not self.is_open:
            return frame

        ui.backdrop(frame, 0, 0, w, h, darkness=0.55)          # dim everything behind the popup
        panel_w, row_h = min(600, w - 24), 84
        panel_h = 80 + len(ROWS) * row_h + 56
        left, top = (w - panel_w) // 2, max(8, (h - panel_h) // 2)
        right_edge, bottom = left + panel_w, top + panel_h
        ui.backdrop(frame, left, top, right_edge, bottom, darkness=0.8)
        ui.text(frame, "Settings", (left + 24, top + 52), 1.2, ui.WHITE, 3)
        self.close_rect = (right_edge - 60, top + 12, right_edge - 12, top + 60)
        ui.text(frame, "X", (right_edge - 46, top + 50), 1.0, ui.GREY, 2)

        self.row_rects = []
        for index, (key, label, description) in enumerate(ROWS):
            y = top + 80 + index * row_h
            self.row_rects.append((key, (left + 12, y, right_edge - 12, y + row_h - 8)))
            ui.backdrop(frame, left + 12, y, right_edge - 12, y + row_h - 8, darkness=0.35)
            ui.text(frame, str(index + 1), (left + 28, y + 34), 0.9, ui.YELLOW, 2)
            ui.text(frame, label, (left + 60, y + 34), 0.9, ui.WHITE, 2)
            ui.text(frame, description, (left + 60, y + 64), 0.6, ui.GREY, 2)
            enabled = getattr(self.store.values, key)
            ui.text(frame, "ON" if enabled else "OFF", (right_edge - 90, y + 48), 0.9,
                    ui.GREEN if enabled else ui.GREY, 2)
        footer = self.store.error or "Click a row or press 1 2 3 to change   ESC = close"
        ui.text(frame, footer, (left + 24, bottom - 22), 0.6, ui.GREY, 2)
        return frame
