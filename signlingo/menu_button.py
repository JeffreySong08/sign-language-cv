"""menu_button.py - the clickable "Main Menu" button shared by both modes.

Draw it every frame (it sits in the footer panel). A left-click inside it, or
ESC, ends the mode's loop, which hands control back to the start menu in main.py.

  install("SignLingo")       -> create the window and start listening for clicks
  draw_menu_button(frame)    -> paint it and remember where it is
  MENU_BUTTON["clicked"]     -> True once the player clicks it
"""

import cv2

MENU_BUTTON_LABEL = "MAIN MENU"
MENU_BUTTON_W, MENU_BUTTON_H = 158, 40
MENU_BUTTON = {"rect": (0, 0, 0, 0), "hover": False, "clicked": False}


def install(window_name):
    """Set up the window and mouse callback before the camera loop starts."""
    MENU_BUTTON["hover"] = False
    MENU_BUTTON["clicked"] = False
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)


def draw_menu_button(frame):
    """Draw the Main Menu button and remember its rectangle for mouse clicks."""
    h, w = frame.shape[:2]
    x1, y1 = w - MENU_BUTTON_W - 32, h - 72
    x2, y2 = x1 + MENU_BUTTON_W, y1 + MENU_BUTTON_H
    MENU_BUTTON["rect"] = (x1, y1, x2, y2)

    hovered = MENU_BUTTON["hover"]
    face, edge = ((63, 63, 70), (255, 255, 255)) if hovered else ((39, 39, 42), (161, 161, 170))
    cv2.rectangle(frame, (x1, y1), (x2, y2), face, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), edge, 1)

    (tw, th), _ = cv2.getTextSize(MENU_BUTTON_LABEL, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, MENU_BUTTON_LABEL,
                (x1 + (MENU_BUTTON_W - tw) // 2, y1 + (MENU_BUTTON_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge, 1, cv2.LINE_AA)
    return frame


def on_mouse(event, x, y, flags, param):
    """Hover highlight and click handling for the Main Menu button."""
    x1, y1, x2, y2 = MENU_BUTTON["rect"]
    inside = x1 <= x <= x2 and y1 <= y <= y2
    MENU_BUTTON["hover"] = inside
    if event == cv2.EVENT_LBUTTONDOWN and inside:
        MENU_BUTTON["clicked"] = True
