"""Step 4: convert MediaPipe's 21 hand landmarks into comparable numbers."""

import math


def hand_to_numbers(hand):
    """Return 63 numbers: normalized x, y, z for each of the 21 dots.

    Pass one detected hand, such as result.hand_landmarks[0]. Each dot
    must have x, y, and z attributes, as MediaPipe landmarks do.

    Subtracting the wrist removes the hand's position. Dividing by the
    distance from the wrist to the middle finger's base removes overall
    size differences. This does not remove changes in hand rotation.
    """
    if len(hand) != 21:
        raise ValueError("Expected exactly 21 hand landmarks.")

    wrist = hand[0]
    middle_finger_base = hand[9]

    # Use a stable part of the palm to measure the hand's size.
    hand_size = math.sqrt(
        (middle_finger_base.x - wrist.x) ** 2
        + (middle_finger_base.y - wrist.y) ** 2
        + (middle_finger_base.z - wrist.z) ** 2
    )
    if hand_size <= 1e-8:
        raise ValueError("Hand size is too small to normalize reliably.")

    numbers = []
    for dot in hand:
        numbers.extend([
            (dot.x - wrist.x) / hand_size,
            (dot.y - wrist.y) / hand_size,
            (dot.z - wrist.z) / hand_size,
        ])

    return numbers
