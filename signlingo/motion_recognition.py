"""Experimental, rule-based J/Z recognition from short fingertip paths.

No training data is required. This is a demo heuristic, not an ASL validator.
Both horizontal directions are accepted for mirrored/left-handed input.
"""

from collections import deque
import math

# Motion tuning: larger path error accepts less exact J/Z shapes.
MAX_PATH_ERROR = 0.24  # Previously 0.16
MIN_PATH_WIDTH = 0.15  # Palm-length units; previously 0.25
MIN_PATH_HEIGHT = 0.25  # Previously 0.40
MIN_ASPECT_RATIO = 0.12
MAX_ASPECT_RATIO = 4.5
MIN_DURATION = 0.2
MAX_DURATION = 3.5


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _resample(points, count=24):
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + _distance(a, b))
    if lengths[-1] < 1e-8:
        return None
    output = []
    segment = 1
    for i in range(count):
        target = lengths[-1] * i / (count - 1)
        while segment < len(points) - 1 and lengths[segment] < target:
            segment += 1
        span = lengths[segment] - lengths[segment - 1]
        fraction = (target - lengths[segment - 1]) / span if span else 0
        a, b = points[segment - 1], points[segment]
        output.append((a[0] + fraction * (b[0] - a[0]),
                       a[1] + fraction * (b[1] - a[1])))
    return output


def _shape(points):
    xs, ys = zip(*points)
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if min(width, height) < 1e-8:
        return None
    return _resample([((x - min(xs)) / width, (y - min(ys)) / height)
                      for x, y in points])


TEMPLATES = {
    # Screen y increases downwards: descend, curl sideways, then up.
    "J": [(0, 0), (0, .72), (.12, .94), (.4, 1), (.75, .88), (1, .62)],
    "Z": [(0, 0), (1, 0), (0, 1), (1, 1)],
}


def match_path(letter, points):
    """Check a path measured in palm-length units, in chronological order."""
    if len(points) < 8:
        return False
    xs, ys = zip(*points)
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    # Require meaningful movement on both axes, not stationary landmark jitter.
    if (width < MIN_PATH_WIDTH or height < MIN_PATH_HEIGHT
            or not MIN_ASPECT_RATIO < width / height < MAX_ASPECT_RATIO):
        return False
    observed = _shape(points)
    template = _shape(TEMPLATES[letter])
    for direction in (1, -1):
        candidate = [(x if direction == 1 else 1 - x, y) for x, y in observed]
        error = sum(_distance(a, b) for a, b in zip(candidate, template)) / len(template)
        if error < MAX_PATH_ERROR:
            return True
    return False


class MotionRecognizer:
    """Feed every frame to update(hand, monotonic_seconds, image_aspect).

    Image-space motion is measured relative to a fixed starting wrist and
    palm size. Subtracting the *current* wrist would erase whole-hand motion.
    A missing hand or changed pose clears the path. Results are briefly held.
    """

    def __init__(self):
        self.reset()
        self.result = None
        self.hold_until = 0

    def reset(self):
        self.history = deque()
        self.letter = None
        self.origin = None
        self.size = None

    @staticmethod
    def _pose(hand, aspect):
        def distance(a, b):
            return math.hypot((hand[a].x - hand[b].x) * aspect,
                              hand[a].y - hand[b].y)

        # Extended fingers have tips farther from the wrist than their PIP joints.
        extended = [distance(tip, 0) > distance(tip - 2, 0) * 1.15
                    for tip in (8, 12, 16, 20)]
        if extended == [False, False, False, True]:
            return "J"
        if extended == [True, False, False, False]:
            return "Z"
        return None

    def update(self, hand, now, aspect=1.0):
        if hand is None:
            self.reset()
            self.result = None
            self.hold_until = 0
            return None
        if now < self.hold_until:
            return self.result
        self.result = None
        letter = self._pose(hand, aspect)
        if letter is None:
            self.reset()
            return None
        if letter != self.letter:
            self.reset()
            self.letter = letter
            self.origin = (hand[0].x * aspect, hand[0].y)
            self.size = math.hypot((hand[9].x - hand[0].x) * aspect,
                                   hand[9].y - hand[0].y)
        if self.size < 1e-6:
            self.reset()
            return None
        tip = hand[20 if letter == "J" else 8]
        point = ((tip.x * aspect - self.origin[0]) / self.size,
                 (tip.y - self.origin[1]) / self.size)
        self.history.append((now, point))
        while self.history and now - self.history[0][0] > MAX_DURATION:
            self.history.popleft()
        # Try several possible starts so an initial pause doesn't hide a gesture.
        history = list(self.history)
        for start in range(0, max(0, len(history) - 7), 2):
            duration = now - history[start][0]
            if MIN_DURATION <= duration <= MAX_DURATION and match_path(letter, [p for _, p in history[start:]]):
                self.result = letter
                self.hold_until = now + .8
                self.reset()
                return letter
        return None
