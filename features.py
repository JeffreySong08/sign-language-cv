"""
features.py - turns one hand into 63 numbers we can compare.

MediaPipe gives us 21 points on the hand (wrist, knuckles, fingertips...).
Their x/y/z values depend on WHERE your hand is in the camera and HOW CLOSE
it is. We don't want that: an "A" is an "A" anywhere on screen. So we:

  1. Measure every point from the wrist (point 0), so the wrist is (0, 0, 0).
  2. Divide by the hand's size (wrist -> middle-finger knuckle, point 9),
     so a close-up hand and a far-away hand give the same numbers.
  3. Return x, y and z for all 21 points in a flat list: 63 numbers.
     (z is MediaPipe's guess at how far each point is from the camera.)

This is Step 4 of the mentor's guide. Use it in other scripts like this:
    from features import hand_to_numbers
    numbers = hand_to_numbers(result.hand_landmarks[0])
"""

import math

WRIST = 0
MIDDLE_KNUCKLE = 9
NUM_FEATURES = 63  # 21 points x (x, y, z)


def hand_to_numbers(hand):
    """hand = result.hand_landmarks[0] from camera.py (21 points with .x, .y, .z)."""
    wrist = hand[WRIST]
    knuckle = hand[MIDDLE_KNUCKLE]

    # Hand size = distance from wrist to middle-finger knuckle (on screen, x and y)
    size = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y)
    if size == 0:          # should never happen, but avoids dividing by zero
        size = 1.0

    numbers = []
    for dot in hand:
        numbers.append((dot.x - wrist.x) / size)
        numbers.append((dot.y - wrist.y) / size)
        numbers.append((dot.z - wrist.z) / size)
    return numbers


# ---------------------------------------------------------------------------
# Quick check. Run:  python features.py
# It makes a fake hand, then the same hand moved and made bigger,
# and shows that both give the SAME 63 numbers (that's the whole point).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace as Point
    import random

    random.seed(1)
    hand = [Point(x=random.random(), y=random.random(), z=random.random()) for _ in range(21)]
    # Same hand, shifted and 2x bigger (like moving closer to the camera)
    moved = [Point(x=p.x * 2 + 0.3, y=p.y * 2 + 0.1, z=p.z * 2 - 0.2) for p in hand]

    a = hand_to_numbers(hand)
    b = hand_to_numbers(moved)

    print("Number of features:", len(a))
    print("First 6 numbers (original hand):", [round(v, 3) for v in a[:6]])
    print("First 6 numbers (moved + bigger):", [round(v, 3) for v in b[:6]])
    same = all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    print("Same numbers?", "YES - features.py works!" if same else "NO - something is wrong")
