"""
tracker.py - hand tracking on a background thread.

Showing a full-resolution frame takes most of a frame's time, so tracking
runs at the same time instead of before it: the mode hands each new camera
picture to the tracker and draws with the newest finished result (at most
one frame old). Each result has a sequence number, so a mode can tell a new
result from one it has already counted.
"""

import threading

import mediapipe as mp


class HandTracker:
    def __init__(self, landmarker):
        self.landmarker = landmarker
        self._lock = threading.Condition()
        self._pending = None          # (rgb, timestamp_ms) waiting to be tracked; only the newest is kept
        self._result = (None, 0)      # (newest result, its sequence number)
        self._error = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="HandTracker", daemon=True)
        self._thread.start()

    def submit(self, rgb, timestamp_ms):
        """Track this picture next (replacing one that hasn't started yet)."""
        with self._lock:
            self._pending = (rgb, timestamp_ms)
            self._lock.notify_all()

    def latest(self, wait=False):
        """(result, sequence number) of the newest finished tracking; (None, 0)
        before the first. wait=True blocks until there is a first result."""
        with self._lock:
            while wait and self._result[1] == 0 and self._error is None:
                self._lock.wait()
            if self._error is not None:
                raise self._error
            return self._result

    def close(self):
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        self._thread.join(timeout=2)

    def _run(self):
        while True:
            with self._lock:
                while self._pending is None and not self._stopping:
                    self._lock.wait()
                if self._stopping:
                    return
                rgb, timestamp_ms = self._pending
                self._pending = None
            try:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(image, timestamp_ms)
            except Exception as error:      # hand it to the mode on its next latest()
                with self._lock:
                    self._error = error
                    self._lock.notify_all()
                return
            with self._lock:
                self._result = (result, self._result[1] + 1)
                self._lock.notify_all()
