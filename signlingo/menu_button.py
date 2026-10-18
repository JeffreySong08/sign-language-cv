"""menu_button.py - the clickable "Main Menu" button shared by both modes.

Draw it every frame (top right, in the same style as the rest of the app).
A left-click inside it, or ESC, ends the mode's loop, which hands control
back to the start menu in main.py.

  install()                  -> start listening for clicks in the SignLingo window
  draw_menu_button(frame)    -> paint it and remember where it is
  MENU_BUTTON["clicked"]     -> True once the player clicks it
"""

import cv2

from signlingo import ui

MENU_BUTTON_LABEL = "Main Menu"
MENU_BUTTON = {"rect": (0, 0, 0, 0), "hover": False, "clicked": False}
_settings_panel = None


def install(window_name=ui.WINDOW, settings_panel=None):
    """Start listening for clicks; call after the window is open."""
    global _settings_panel
    _settings_panel = settings_panel
    MENU_BUTTON["rect"] = (0, 0, 0, 0)
    MENU_BUTTON["hover"] = False
    MENU_BUTTON["clicked"] = False
    cv2.setMouseCallback(window_name, on_mouse)


def draw_menu_button(frame):
    """Draw the Main Menu button and remember its rectangle for mouse clicks."""
    w = ui.points(frame)[0]
    label_w = ui.text_width(MENU_BUTTON_LABEL, 0.8, 2)
    x2, y1 = w - 8, 8
    x1, y2 = x2 - label_w - 40, y1 + 52            # at least 44 px tall to click easily
    MENU_BUTTON["rect"] = (x1, y1, x2, y2)

    hovered = MENU_BUTTON["hover"]
    ui.backdrop(frame, x1, y1, x2, y2, darkness=0.75 if hovered else 0.6)
    ui.text(frame, MENU_BUTTON_LABEL, (x1 + 20, y1 + 35), 0.8, ui.YELLOW if hovered else ui.WHITE, 2)
    return frame


def on_mouse(event, x, y, flags, param):
    """Settings consume clicks first so the menu cannot fire behind the popup."""
    x, y = ui.to_points(x, y)        # OpenCV reports real pixels; layout is in points
    if _settings_panel is not None and _settings_panel.on_mouse(event, x, y, flags, param):
        MENU_BUTTON["hover"] = False
        return
    x1, y1, x2, y2 = MENU_BUTTON["rect"]
    inside = x1 <= x <= x2 and y1 <= y <= y2
    MENU_BUTTON["hover"] = inside
    if event == cv2.EVENT_LBUTTONDOWN and inside:
        MENU_BUTTON["clicked"] = True
