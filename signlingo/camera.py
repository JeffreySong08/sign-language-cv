"""
camera.py - the webcam, opened once and shared.

The menu opens the camera (to check permission) and leaves it open, so a
mode starts right away instead of waiting about a second for it to reopen.

A background thread keeps reading the camera, so read() returns the newest
picture: a frame that takes a little longer than the camera's 1/30 s then
doesn't have to wait a whole extra frame for the next picture.
"""

import threading

import cv2

_cap = None


class LatestFrameCamera:
    """Wraps cv2.VideoCapture; read() returns the newest picture not yet returned."""

    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)
        self._lock = threading.Condition()
        self._frame = None
        self._number = 0        # pictures read from the camera so far
        self._returned = 0      # the last picture number read() returned
        self._stopped = not self._cap.isOpened()
        if not self._stopped:
            threading.Thread(target=self._run, name="Camera", daemon=True).start()

    def _run(self):
        while True:
            ok, frame = self._cap.read()
            with self._lock:
                if self._stopped:
                    return
                if not ok:
                    self._stopped = True
                    self._lock.notify_all()
                    return
                self._frame, self._number = frame, self._number + 1
                self._lock.notify_all()

    def read(self):
        """(ok, newest picture), waiting only if it was already returned."""
        with self._lock:
            while self._number == self._returned and not self._stopped:
                self._lock.wait(timeout=2)
                if self._number == self._returned:
                    return False, None      # the camera stopped sending pictures
            if self._number == self._returned:
                return False, None
            self._returned = self._number
            return True, self._frame

    def isOpened(self):
        return not self._stopped

    def release(self):
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        self._cap.release()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)


def open_camera():
    """The open webcam, opening it if needed."""
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = LatestFrameCamera(0)
    return _cap


def close_camera():
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
