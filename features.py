"""
features.py - the shared "recipe" for turning a hand into numbers,
plus the letter guesser. Copied from Karen's camera.py so every script
(camera.py, collect_data.py, practice.py) stays in agreement.

If Karen changes hand_to_numbers or guess_letter in camera.py,
copy the change here too, or old and new examples won't match.

  hand_to_numbers(hand)           -> 42 numbers for one hand
  guess_letter(numbers, examples) -> (letter, confidence) or (None, 0.0)
  load_all_examples()             -> every example in data/examples_*.json
"""

import glob
import json
import math
import os

KNN_K = 5
MAX_NEIGHBOR_DISTANCE = 15.0  # if even the best match is further than this, say "unknown"

# Fingertips: thumb (4), index (8), middle (12), ring (16), pinky (20)
FINGERTIPS = {4, 8, 12, 16, 20}
NUM_NUMBERS = 42  # 21 points x (x, y)
MOTION_LETTERS = ("J", "Z")  # recognized from movement (motion_recognition.py)
DATA_DIR = "data"            # all examples_*.json files live here


def load_all_examples(verbose=True):
    """Load every data/examples_*.json made with this recipe (skips broken files)."""
    examples = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "examples_*.json"))):
        try:
            with open(path) as f:
                found = [e for e in json.load(f)
                         if len(e.get("numbers", [])) == NUM_NUMBERS
                         and e.get("letter") not in MOTION_LETTERS]
        except (ValueError, AttributeError, TypeError):
            print(f"Skipping {path}: it isn't a readable examples file")
            continue
        if verbose:
            print(f"Loaded {len(found)} examples from {path}")
        examples += found
    return examples


def hand_to_numbers(hand):
    """hand = result.hand_landmarks[0] (21 points with .x and .y)."""
    wrist = hand[0]
    knuckle = hand[9]  # base knuckle of middle finger
    size = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y) or 1e-6
    numbers = []
    for idx, dot in enumerate(hand):
        # Fingertips count 2.5x, because finger shape matters most for letters
        weight = 2.5 if idx in FINGERTIPS else 1.0
        numbers.append(((dot.x - wrist.x) / size) * weight)
        numbers.append(((dot.y - wrist.y) / size) * weight)
    return numbers


def guess_letter(numbers, examples, k=KNN_K):
    """Find the k closest saved examples and let them vote on the letter."""
    distances = []
    for item in examples:
        if item["letter"] in MOTION_LETTERS:
            continue  # J/Z come from motion, never from a single pose
        if len(item["numbers"]) != len(numbers):
            continue  # skip examples made with an old recipe
        dist = sum((a - b) ** 2 for a, b in zip(numbers, item["numbers"]))
        distances.append((dist, item["letter"]))
    if not distances:
        return None, 0.0

    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    if top_k[0][0] > MAX_NEIGHBOR_DISTANCE:
        return None, 0.0

    votes = {}
    for _, letter in top_k:
        votes[letter] = votes.get(letter, 0) + 1
    best = max(votes, key=votes.get)
    return best, votes[best] / len(top_k)


# Quick check:  python features.py
if __name__ == "__main__":
    from types import SimpleNamespace as Point
    import random

    random.seed(1)
    hand = [Point(x=random.random(), y=random.random()) for _ in range(21)]
    moved = [Point(x=p.x * 2 + 0.3, y=p.y * 2 + 0.1) for p in hand]
    a, b = hand_to_numbers(hand), hand_to_numbers(moved)
    print("Number of features:", len(a))
    same = all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    print("Same numbers when moved/bigger?", "YES" if same else "NO")
    print("Guess:", guess_letter(b, [{"letter": "A", "numbers": a}]))
