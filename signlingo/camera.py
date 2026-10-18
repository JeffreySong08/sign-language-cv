"""
camera.py - the webcam, opened once and shared.

The menu opens the camera (to check permission) and leaves it open, so a
mode starts right away instead of waiting about a second for it to reopen.
"""

import cv2

_cap = None


def open_camera():
    """The open webcam, opening it if needed."""
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(0)
    return _cap


def close_camera():
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
