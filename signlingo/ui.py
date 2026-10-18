"""
ui.py - the one SignLingo window and the look every screen shares.

Every screen (menu, letter practice, word challenge) is drawn at the
screen's own size and shown full screen in the same window, in the Single
Letter Practice text style: big white text, a yellow "I see", a green
hold bar, grey help text.

Sizes and positions are in screen points (what macOS calls the screen size,
e.g. 1470 x 924). Frames have SCALE real pixels per point (2 on a Retina
screen), so text and lines are drawn at full resolution with sharp edges.
"""

import ctypes
import sys

import cv2
import numpy as np

from signlingo.paths import SIGNS_DIR

WINDOW = "SignLingo"
_size = None
SCALE = 1                      # real pixels per point; set by screen_size()

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)        # colors are BGR
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
GREY = (200, 200, 200)
DIM = (170, 170, 170)        # letters still to come
TIP = (255, 200, 0)            # light blue: J/Z movement tips, focus info
BACKGROUND = (30, 30, 30)


def screen_size():
    """(width, height) a full-screen window gets, in points. Every screen is
    drawn at this size, so full screen has no empty bars. 1280x720 if unknown."""
    global _size, SCALE
    if _size is None:
        found = _mac_screen_size() if sys.platform == "darwin" else None
        if found:
            _size, SCALE = found[:2], found[2]
        else:
            _size, SCALE = (1280, 720), 1
    return _size


def pixel_size():
    """(width, height) of every frame, in real pixels."""
    w, h = screen_size()
    return round(w * SCALE), round(h * SCALE)


def px(value):
    """Points -> real pixels."""
    return int(round(value * SCALE))


def points(frame):
    """(width, height) of a frame in points."""
    h, w = frame.shape[:2]
    return round(w / SCALE), round(h / SCALE)


def to_points(x, y):
    """Mouse position (real pixels of the frame) -> points."""
    return x / SCALE, y / SCALE


def blank():
    """An empty screen-sized frame in the background color."""
    w, h = pixel_size()
    frame = np.empty((h, w, 3), np.uint8)
    frame[:] = BACKGROUND
    return frame


def _mac_screen_size():
    """Main screen size from macOS in points, minus the strip beside the camera
    notch (full-screen windows sit below it), and its pixels per point."""
    class Rect(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                    ("width", ctypes.c_double), ("height", ctypes.c_double)]

    class Insets(ctypes.Structure):
        _fields_ = [("top", ctypes.c_double), ("left", ctypes.c_double),
                    ("bottom", ctypes.c_double), ("right", ctypes.c_double)]

    try:
        ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        call = lambda result, obj, name: ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p)(send)(
            obj, objc.sel_registerName(name))
        screen = call(ctypes.c_void_p, objc.objc_getClass(b"NSScreen"), b"mainScreen")
        if not screen:
            return None
        frame = call(Rect, screen, b"frame")
        notch = call(Insets, screen, b"safeAreaInsets")   # all zero without a notch
        scale = call(ctypes.c_double, screen, b"backingScaleFactor")
        width = frame.width - notch.left - notch.right
        height = frame.height - notch.top - notch.bottom
        scale = int(scale) if scale >= 1 else 1
        return (int(width), int(height), scale) if width > 0 and height > 0 else None
    except (OSError, AttributeError):
        return None


def open_window():
    """Create the SignLingo window (once) and make it fill the screen."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def fit(frame):
    """Scale and center-crop a camera frame to the screen's size (real pixels)."""
    w, h = pixel_size()
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    frame = cv2.resize(frame, (round(fw * scale), round(fh * scale)))
    fh, fw = frame.shape[:2]
    x, y = (fw - w) // 2, (fh - h) // 2
    return frame[y:y + h, x:x + w]


def for_detection(frame):
    """A smaller copy for hand tracking: the full-resolution frame only
    matters for drawing, and tracking a smaller picture is faster."""
    if SCALE == 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (round(w / SCALE), round(h / SCALE)), interpolation=cv2.INTER_AREA)


def backdrop(frame, x1, y1, x2, y2, darkness=0.6):
    """See-through dark box, so text stays readable on bright video."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = px(x1), px(y1), px(x2), px(y2)
    box = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    cv2.convertScaleAbs(box, dst=box, alpha=1 - darkness)    # fast whole-number darkening


def top_backdrop(frame, widest, bottom):
    """Box behind the top-left text block; widest = width of its longest line."""
    backdrop(frame, 8, 8, 20 + max(widest, 300) + 16, bottom)


_signs = {}


def sign_picture(letter, height):
    """The drawn hand for a letter (assets/signs/<letter>.png, with
    transparency), scaled to this height, ready to blend:
    (color x opacity, 1 - opacity), both 16-bit. None if there's no picture."""
    key = (letter, height)
    if key not in _signs:
        rgba = cv2.imread(str(SIGNS_DIR / f"{letter}.png"), cv2.IMREAD_UNCHANGED)
        picture = None
        if rgba is not None and rgba.shape[2] == 4:
            h, w = rgba.shape[:2]
            rgba = cv2.resize(rgba, (round(w * height / h), height), interpolation=cv2.INTER_AREA)
            alpha = rgba[:, :, 3:].astype(np.uint16)
            picture = (rgba[:, :, :3] * alpha, np.repeat(255 - alpha, 3, axis=2))
        _signs[key] = picture
    return _signs[key]


def sign_card(frame, letter, height=230):
    """How to sign the letter: the drawn hand in the bottom-right corner,
    straight on the video (its background is transparent)."""
    picture = sign_picture(letter, px(height))
    if picture is None:
        return
    color, see_through = picture
    ph, pw = color.shape[:2]
    h, w = frame.shape[:2]
    x2, y2 = w - px(16), h - px(16)
    spot = frame[y2 - ph:y2, x2 - pw:x2]
    spot[:] = ((color + spot * see_through) // 255).astype(np.uint8)   # whole-number blend


def show(frame):
    cv2.imshow(WINDOW, frame)


def text(frame, words, pos, scale, color, thickness):
    """Text at pos (points), sized in points, drawn with smooth edges."""
    cv2.putText(frame, words, (px(pos[0]), px(pos[1])), FONT, scale * SCALE, color,
                max(1, px(thickness)), cv2.LINE_AA)


def text_width(words, scale, thickness):
    """Width of the text in points."""
    return cv2.getTextSize(words, FONT, scale * SCALE, max(1, px(thickness)))[0][0] / SCALE


def text_height(words, scale, thickness):
    return cv2.getTextSize(words, FONT, scale * SCALE, max(1, px(thickness)))[0][1] / SCALE


def centered(frame, words, y, scale, color, thickness, boxed=False):
    width = text_width(words, scale, thickness)
    x = (points(frame)[0] - width) // 2
    if boxed:
        height = text_height(words, scale, thickness)
        backdrop(frame, x - 30, y - height - 30, x + width + 30, y + 30)
    text(frame, words, (x, y), scale, color, thickness)


def hold_bar(frame, streak, frames_needed):
    """Green bar that fills as you hold the right sign."""
    bar = 300 * min(streak / frames_needed, 1.0)
    cv2.rectangle(frame, (px(20), px(95)), (px(320), px(115)), (170, 170, 170), px(2), cv2.LINE_AA)
    if bar > 0:
        cv2.rectangle(frame, (px(20), px(95)), (px(20 + bar), px(115)), GREEN, -1)


def hand_dots(frame, hand):
    h, w = frame.shape[:2]
    for p in hand:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), px(4), GREEN, -1, cv2.LINE_AA)


def help_line(frame, words, above=None):
    """Grey help text at the bottom left on a see-through box, with an
    optional line above it in the same box (focus info)."""
    h = points(frame)[1]
    widest = max(text_width(words, 0.6, 2), text_width(above, 0.8, 2) if above else 0)
    backdrop(frame, 8, h - (75 if above else 45), 20 + widest + 16, h - 8)
    if above:
        text(frame, above, (20, h - 50), 0.8, TIP, 2)
    text(frame, words, (20, h - 20), 0.6, GREY, 2)


def motion_tip(letter):
    return "draw a J with your pinky" if letter == "J" else "draw a Z with your index finger"
