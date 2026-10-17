"""Run without a webcam or MediaPipe: python -m unittest test_motion_recognition"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from motion_recognition import MotionRecognizer, TEMPLATES, _resample, match_path


def make_hand(letter, dx=0, dy=0):
    hand = [SimpleNamespace(x=dx, y=dy, z=0) for _ in range(21)]
    hand[9].y = dy - 1
    for tip in (8, 12, 16, 20):
        hand[tip - 2].y = dy - .8
        hand[tip].y = dy - (.4 if tip != (20 if letter == "J" else 8) else 1.5)
    return hand


class MotionTests(unittest.TestCase):
    def test_paths_and_mirrors(self):
        for letter, template in TEMPLATES.items():
            path = _resample(template, 30)
            self.assertTrue(match_path(letter, path))
            self.assertTrue(match_path(letter, [(-x, y) for x, y in path]))
            self.assertFalse(match_path(letter, list(reversed(path))))
            self.assertFalse(match_path(letter, [(x * .01, y * .01) for x, y in path]))

    def test_smaller_and_distorted_paths(self):
        paths = {
            'J': [(0, 0), (.2, .65), (.1, 1), (.6, .95), (1, .8)],
            'Z': [(0, 0), (1, .25), (.15, .8), (.9, 1)],
        }
        for letter, points in paths.items():
            self.assertTrue(match_path(letter, _resample(points, 30)))
            smaller = [(x * .2, y * .3) for x, y in TEMPLATES[letter]]
            self.assertTrue(match_path(letter, _resample(smaller, 30)))

    def test_stationary_and_straight_paths_rejected(self):
        for letter in TEMPLATES:
            for path in ([(0, 0)] * 30, [(i / 30, i / 30) for i in range(30)]):
                self.assertFalse(match_path(letter, path))

    def test_pose_gate(self):
        for letter in TEMPLATES:
            self.assertEqual(MotionRecognizer._pose(make_hand(letter), 1), letter)

    def test_live_sequence_hold_and_missing_hand(self):
        for letter, template in TEMPLATES.items():
            tracker = MotionRecognizer()
            results = []
            # Translate the entire hand: current-wrist subtraction would lose this motion.
            for i, (x, y) in enumerate(_resample(template, 30)):
                results.append(tracker.update(make_hand(letter, x, y), i * .04))
            self.assertIn(letter, results)
            self.assertEqual(tracker.update(make_hand(letter), 1.2), letter)
            self.assertIsNone(tracker.update(None, 1.3))

    def test_wrong_pose_and_tracking_gap(self):
        tracker = MotionRecognizer()
        with patch.object(tracker, '_pose', return_value=None):
            self.assertIsNone(tracker.update(make_hand('J'), 0))
        tracker.update(make_hand('J'), 0)
        tracker.update(None, .1)
        self.assertFalse(tracker.history)


if __name__ == '__main__':
    unittest.main()
