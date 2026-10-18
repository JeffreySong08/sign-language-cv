"""Settings, both modes, and window-close behavior without using a webcam (from PR #1)."""

from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import main as launcher
from signlingo import camera, menu_button, ui
from signlingo.settings import SettingsStore
from signlingo.settings_ui import SettingsPanel, limit_frame_rate, window_is_open


class SettingsTests(unittest.TestCase):
    def test_defaults_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            store = SettingsStore(path)
            self.assertFalse(store.values.low_power)
            self.assertTrue(store.values.show_landmarks)
            self.assertTrue(store.values.selfie)
            self.assertTrue(store.values.tutorial_mode)
            for key in ('low_power', 'show_landmarks', 'selfie', 'tutorial_mode'):
                store.toggle(key)
            restored = SettingsStore(path)
            self.assertTrue(restored.values.low_power)
            self.assertFalse(restored.values.show_landmarks)
            self.assertFalse(restored.values.selfie)
            self.assertFalse(restored.values.tutorial_mode)
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_invalid_values_do_not_enable_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'low_power': 'false', 'selfie': False,
                                        'show_landmarks': 0, 'tutorial_mode': 'false'}))
            store = SettingsStore(path)
            self.assertFalse(store.values.low_power)
            self.assertFalse(store.values.selfie)
            self.assertTrue(store.values.show_landmarks)
            self.assertTrue(store.values.tutorial_mode)
            with self.assertRaises(ValueError):
                store.toggle('invalid')

    def test_older_settings_keep_tutorial_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'low_power': True, 'show_landmarks': False, 'selfie': False}))
            store = SettingsStore(path)
            self.assertTrue(store.values.tutorial_mode)
            self.assertTrue(store.values.low_power)
            self.assertFalse(store.values.show_landmarks)
            self.assertFalse(store.values.selfie)

    def test_save_failure_keeps_session_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / 'settings.json')
            with patch.object(Path, 'mkdir', side_effect=PermissionError), self.assertLogs(level='ERROR'):
                store.toggle('low_power')
            self.assertTrue(store.values.low_power)
            self.assertTrue(store.error)

    def test_keyboard_and_mouse_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / 'settings.json')
            panel = SettingsPanel(store)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            panel.draw(frame)
            x1, y1, x2, y2 = panel.gear_rect
            panel.on_mouse(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2)
            self.assertTrue(panel.is_open)
            panel.draw(frame)
            key, rect = panel.row_rects[0]
            panel.on_mouse(cv2.EVENT_LBUTTONDOWN, rect[0] + 4, rect[1] + 4)
            self.assertTrue(store.values.low_power)
            self.assertTrue(panel.handle_key(ord('2')))
            self.assertFalse(store.values.show_landmarks)
            self.assertTrue(panel.handle_key(ord('3')))
            self.assertFalse(store.values.selfie)
            self.assertTrue(panel.handle_key(ord('4')))
            self.assertFalse(store.values.tutorial_mode)
            key, rect = panel.row_rects[3]
            self.assertEqual(key, 'tutorial_mode')
            panel.on_mouse(cv2.EVENT_LBUTTONDOWN, rect[0] + 4, rect[1] + 4)
            self.assertTrue(store.values.tutorial_mode)
            self.assertTrue(SettingsStore(store.path).values.tutorial_mode)
            self.assertTrue(panel.handle_key(27))
            self.assertFalse(panel.is_open)
            self.assertFalse(panel.handle_key(ord('1')))
            self.assertFalse(panel.handle_key(ord('4')))

    def test_four_settings_rows_fit_on_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            for width, height in ((640, 480), (1280, 720), (1470, 924)):
                with self.subTest(size=(width, height)):
                    panel = SettingsPanel(SettingsStore(Path(directory) / 'settings.json'))
                    panel.is_open = True
                    panel.draw(np.zeros((height, width, 3), dtype=np.uint8))
                    self.assertEqual(len(panel.row_rects), 4)
                    for key, (x1, y1, x2, y2) in panel.row_rects:
                        self.assertGreaterEqual(x1, 0)
                        self.assertGreaterEqual(y1, 0)
                        self.assertLess(x2, width)
                        self.assertLess(y2, height)

    def test_gear_sits_left_of_main_menu_button(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = SettingsPanel(SettingsStore(Path(directory) / 'settings.json'))
            frame = np.zeros((924, 1470, 3), dtype=np.uint8)
            menu_button.draw_menu_button(frame)
            panel.draw(frame, right=menu_button.MENU_BUTTON['rect'][0])
            self.assertLess(panel.gear_rect[2], menu_button.MENU_BUTTON['rect'][0])

    def test_popup_consumes_main_menu_clicks(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = SettingsPanel(SettingsStore(Path(directory) / 'settings.json'))
            panel.is_open = True
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(cv2, 'setMouseCallback'):
                menu_button.install('SignLingo', panel)
            menu_button.draw_menu_button(frame)
            panel.draw(frame)
            x1, y1, _, _ = menu_button.MENU_BUTTON['rect']
            menu_button.on_mouse(cv2.EVENT_LBUTTONDOWN, x1 + 5, y1 + 5, 0, None)
            self.assertFalse(menu_button.MENU_BUTTON['clicked'])

    def test_frame_rate_limiter(self):
        with patch('signlingo.settings_ui.time.monotonic', return_value=10.04), \
                patch('signlingo.settings_ui.time.sleep') as sleep:
            limit_frame_rate(10, False)
            sleep.assert_not_called()
            limit_frame_rate(10, True)
            self.assertAlmostEqual(sleep.call_args.args[0], 1 / 12 - .04)


def menu_window_patches(stack):
    for method in ('namedWindow', 'setWindowProperty', 'setMouseCallback', 'imshow', 'destroyAllWindows'):
        stack.enter_context(patch.object(cv2, method))


class LifecycleTests(unittest.TestCase):
    def test_closed_and_destroyed_windows_are_detected(self):
        with patch.object(cv2, 'getWindowProperty', return_value=0):
            self.assertFalse(window_is_open())
        with patch.object(cv2, 'getWindowProperty', side_effect=cv2.error('window gone')):
            self.assertFalse(window_is_open())
        with patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(window_is_open())

    def test_menu_close_does_not_wait_for_a_key(self):
        with ExitStack() as stack:
            menu_window_patches(stack)
            wait = stack.enter_context(patch.object(cv2, 'waitKey', return_value=255))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=0))
            self.assertIsNone(launcher.choose_mode())
            wait.assert_called_with(30)

    def test_settings_keys_cannot_accidentally_start_mode(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            panel = SettingsPanel(SettingsStore(Path(directory) / 'settings.json'))
            stack.enter_context(patch.object(launcher, 'SettingsPanel', return_value=panel))
            menu_window_patches(stack)
            # First draw; open settings, toggle 1 and 4, close, start mode 2.
            stack.enter_context(patch.object(cv2, 'waitKey',
                                            side_effect=[255, ord(','), ord('1'), ord('4'), 27, ord('2')]))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=1))
            self.assertEqual(launcher.choose_mode(), 'signlingo.words')
            self.assertTrue(panel.store.values.low_power)
            self.assertFalse(panel.store.values.tutorial_mode)

    def test_menu_options_can_be_clicked(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            panel = SettingsPanel(SettingsStore(Path(directory) / 'settings.json'))
            stack.enter_context(patch.object(launcher, 'SettingsPanel', return_value=panel))
            menu_window_patches(stack)
            callback = {}
            cv2.setMouseCallback.side_effect = lambda name, fn: callback.setdefault('fn', fn)
            w, h = ui.screen_size()
            def click_spell_words(delay):     # click the middle of the "2 Spell Words" line
                if delay == 30:
                    callback['fn'](cv2.EVENT_LBUTTONDOWN, ui.px(w // 2), ui.px(h // 2 + 110), 0, None)   # real pixels
                return 255
            stack.enter_context(patch.object(cv2, 'waitKey', side_effect=click_spell_words))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=1))
            self.assertEqual(launcher.choose_mode(), 'signlingo.words')

    def test_native_mode_close_quits_instead_of_reopening_menu(self):
        with patch.object(launcher, 'choose_mode', return_value='signlingo.words') as menu, \
                patch.object(launcher, 'wait_for_camera', return_value=True), \
                patch.object(launcher.runpy, 'run_module', side_effect=SystemExit(0)), \
                patch.object(cv2, 'destroyAllWindows'):
            with self.assertRaises(SystemExit):
                launcher.main()
            menu.assert_called_once()


class ModeIntegrationTests(unittest.TestCase):
    def run_one_frame(self, mode, store, close=False, keys=None):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, 0] = [0, 0, 255]                       # red left edge, to see mirroring
        hand = [SimpleNamespace(x=.01 * i, y=.02 * i) for i in range(21)]
        detector = Mock()
        detector.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand])
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame.copy())
        camera.close_camera()
        with ExitStack() as stack:
            stack.enter_context(patch('signlingo.settings.get_settings', return_value=store))
            stack.enter_context(patch.object(sys, 'argv', ['main.py']))
            stack.enter_context(patch.object(cv2, 'VideoCapture', return_value=cap))
            from mediapipe.tasks.python import vision
            stack.enter_context(patch.object(vision.HandLandmarker, 'create_from_options', return_value=detector))
            for method in ('namedWindow', 'setWindowProperty', 'setMouseCallback', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch.object(cv2, method))
            stack.enter_context(patch.object(cv2, 'waitKey', side_effect=keys,
                                            return_value=255 if close else 27))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=0 if close else 1))
            tutorial_states = []
            def recognize(*args):
                tutorial_states.append(store.values.tutorial_mode)
                return 'B', .8
            guess = stack.enter_context(patch('signlingo.features.guess_letter', side_effect=recognize))
            dots = stack.enter_context(patch.object(ui, 'hand_dots'))
            pictures = stack.enter_context(patch.object(ui, 'sign_card'))
            if close:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_module('signlingo.' + mode, run_name='__main__')
                self.assertEqual(stopped.exception.code, 0)
            else:
                runpy.run_module('signlingo.' + mode, run_name='__main__')
            self.assertEqual(guess.call_count, len(keys) if keys is not None else 1)
            self.assertEqual(dots.called, store.values.show_landmarks)
            self.assertEqual(pictures.call_count, sum(tutorial_states) if mode == 'practice' else 0)
            rgb = detector.detect_for_video.call_args.args[0].numpy_view()
            column = -1 if store.values.selfie else 0
            np.testing.assert_array_equal(rgb[0, column], [255, 0, 0])
        detector.close.assert_called_once()
        camera.close_camera()          # stop the camera's reading thread

    def test_tutorial_only_controls_letter_pictures(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / 'settings.json')
            for enabled in (True, False):
                with self.subTest(tutorial=enabled):
                    store.values.tutorial_mode = enabled
                    self.run_one_frame('practice', store)

    def test_tutorial_can_change_during_practice(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / 'settings.json')
            self.run_one_frame('practice', store, keys=[ord(','), ord('4'), 27, 27])
            self.assertFalse(store.values.tutorial_mode)
            self.assertFalse(SettingsStore(store.path).values.tutorial_mode)

    def test_options_and_native_close_work_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / 'settings.json')
            for mode in ('practice', 'words'):
                for enabled in (False, True):
                    with self.subTest(mode=mode, landmarks=enabled, selfie=enabled):
                        store.values.show_landmarks = enabled
                        store.values.selfie = enabled
                        self.run_one_frame(mode, store)
                with self.subTest(mode=mode, close=True):
                    self.run_one_frame(mode, store, close=True)


if __name__ == '__main__':
    unittest.main()
