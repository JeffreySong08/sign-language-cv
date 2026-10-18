"""The camera-reading and hand-tracking threads, without a webcam."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from signlingo import camera
from signlingo.tracker import HandTracker


class FakeCapture:
    """Hands out numbered pictures, one each time read() is called."""
    def __init__(self, *args):
        self.count = 0
        self.released = threading.Event()

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.005)
        self.count += 1
        return True, np.full((2, 2, 3), self.count % 256, np.uint8)

    def release(self):
        self.released.set()


class CameraTests(unittest.TestCase):
    def test_read_returns_newest_picture_and_never_repeats(self):
        with patch.object(camera.cv2, 'VideoCapture', FakeCapture):
            cam = camera.LatestFrameCamera()
            try:
                seen = []
                for _ in range(5):
                    ok, frame = cam.read()
                    self.assertTrue(ok)
                    seen.append(int(frame[0, 0, 0]))
                    time.sleep(0.02)            # slower than the camera: pictures in between are skipped
                self.assertEqual(len(set(seen)), len(seen))
                self.assertEqual(seen, sorted(seen))
            finally:
                cam.release()
            self.assertFalse(cam.isOpened())


class TrackerTests(unittest.TestCase):
    def test_results_are_numbered_and_first_can_be_awaited(self):
        landmarker = SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(timestamp=ts))
        tracker = HandTracker(landmarker)
        try:
            self.assertEqual(tracker.latest(), (None, 0))
            tracker.submit(np.zeros((4, 4, 3), np.uint8), 10)
            result, seq = tracker.latest(wait=True)
            self.assertEqual((result.timestamp, seq), (10, 1))
            tracker.submit(np.zeros((4, 4, 3), np.uint8), 20)
            for _ in range(200):
                if tracker.latest()[1] == 2:
                    break
                time.sleep(0.005)
            self.assertEqual(tracker.latest()[1], 2)
        finally:
            tracker.close()

    def test_tracking_errors_reach_the_mode(self):
        def fail(image, ts):
            raise RuntimeError('model broke')
        tracker = HandTracker(SimpleNamespace(detect_for_video=fail))
        try:
            tracker.submit(np.zeros((4, 4, 3), np.uint8), 1)
            with self.assertRaises(RuntimeError):
                tracker.latest(wait=True)
        finally:
            tracker.close()


if __name__ == '__main__':
    unittest.main()
