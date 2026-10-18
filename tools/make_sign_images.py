"""
make_sign_images.py - cut an ASL alphabet chart into one picture per letter.

Run:  python -m tools.make_sign_images path/to/alphabet_chart.png

Expects a chart like the "Hand Drawn Sign Language Alphabet": 26 drawn,
skin-colored hands on a white background in rows (A-G, H-N, O-U, V-Z), each
with its letter printed above it. Saves assets/signs/A.png ... Z.png with a
transparent background and WITHOUT the printed letter (movement arrows, like
J's hook and Z's zigzag, are kept). Also saves assets/signs/_preview.png to
check the result at a glance.
"""

import sys

import cv2
import numpy as np

from signlingo.paths import SIGNS_DIR

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_HAND_AREA = 1500   # px of skin color; smaller blobs are parts, not hands
MAX_ARROW_AREA = 1500  # px; a movement arrow is a small mark, never a whole hand
ARROW_REACH = 90       # px beside a hand where a movement arrow may sit
PADDING = 6


def skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (5, 25, 170), (30, 120, 255)) > 0


def find_hands(image):
    """Skin-colored blobs, one per hand, in reading order (row by row)."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(skin_mask(image).astype(np.uint8))
    hands = [tuple(int(v) for v in s[:4]) for s in stats[1:] if s[4] > MIN_HAND_AREA]
    if len(hands) != len(LETTERS):
        raise SystemExit(f"Found {len(hands)} hands, expected 26. Is this the right kind of chart?")
    hands.sort(key=lambda b: b[1] + b[3] / 2)           # top to bottom
    rows, row = [], [hands[0]]
    for b in hands[1:]:                                 # a new row starts when a hand is clearly lower
        if b[1] + b[3] / 2 - (row[-1][1] + row[-1][3] / 2) > 100:
            rows.append(row)
            row = []
        row.append(b)
    rows.append(row)
    return [b for r in rows for b in sorted(r, key=lambda b: b[0])]


def hand_parts(drawing_labels, skin, hand):
    """The drawn parts (fill + outline) that hold this hand's skin."""
    hx, hy, hw, hh = hand
    hand_skin = skin[hy:hy + hh, hx:hx + hw]
    return set(np.unique(drawing_labels[hy:hy + hh, hx:hx + hw][hand_skin])) - {0}


def find_arrows(drawing_labels, stats, skin, parts):
    """Movement arrows (J's hook, Z's zigzag): small marks without skin, level
    with a hand rather than above it like the letters. Each goes to the
    nearest hand beside it. Returns {hand index: set of parts}."""
    taken = set().union(*parts)
    spans = [(min(stats[k][1] for k in p), max(stats[k][1] + stats[k][3] for k in p),
              min(stats[k][0] for k in p), max(stats[k][0] + stats[k][2] for k in p)) for p in parts]
    arrows = {i: set() for i in range(len(parts))}
    for k, (x, y, w, h, area) in enumerate(stats):
        if k == 0 or k in taken or area > MAX_ARROW_AREA:
            continue
        if skin[drawing_labels == k].sum() > 50:     # filled with skin: part of a hand
            continue
        center_y = y + h / 2
        gaps = [(max(left - (x + w), x - right, 0), i)
                for i, (top, bottom, left, right) in enumerate(spans)
                if top + 10 < center_y < bottom]       # level with the hand, not the letter above it
        gaps = [g for g in gaps if g[0] < ARROW_REACH]
        if gaps:
            arrows[min(gaps)[1]].add(k)
    return arrows


def cut_out(image, drawing_labels, keep):
    """The chosen drawn parts on a transparent background, cropped tight."""
    mask = np.isin(drawing_labels, list(keep)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))           # keep the soft outline edge
    ys, xs = np.nonzero(mask)
    x1, y1 = max(xs.min() - PADDING, 0), max(ys.min() - PADDING, 0)
    x2, y2 = xs.max() + PADDING + 1, ys.max() + PADDING + 1

    crop = image[y1:y2, x1:x2]
    # Transparent where the paper is white, solid where it is drawn, soft in between
    whiteness = crop.min(axis=2).astype(np.float32)
    alpha = np.clip((250 - whiteness) * 255 / 50, 0, 255) * mask[y1:y2, x1:x2]
    return np.dstack([crop, alpha.astype(np.uint8)])


def preview(pictures):
    """All 26 cut-outs on a grey sheet, labelled, to check nothing is missing."""
    cell = 180
    sheet = np.full((cell * 4, cell * 7, 3), 90, np.uint8)
    for i, (letter, rgba) in enumerate(pictures):
        h, w = rgba.shape[:2]
        scale = (cell - 30) / max(h, w)
        rgba = cv2.resize(rgba, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = rgba.shape[:2]
        x0, y0 = (i % 7) * cell + (cell - w) // 2, (i // 7) * cell + 22
        a = rgba[:, :, 3:] / 255.0
        region = sheet[y0:y0 + h, x0:x0 + w]
        region[:] = (rgba[:, :, :3] * a + region * (1 - a)).astype(np.uint8)
        cv2.putText(sheet, letter, ((i % 7) * cell + 6, (i // 7) * cell + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return sheet


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Can't read {sys.argv[1]}")
    drawn = (image.min(axis=2) < 230).astype(np.uint8)            # anything not white paper
    skin = skin_mask(image)
    _, drawing_labels, stats, _ = cv2.connectedComponentsWithStats(drawn)
    SIGNS_DIR.mkdir(parents=True, exist_ok=True)
    parts = [hand_parts(drawing_labels, skin, hand) for hand in find_hands(image)]
    arrows = find_arrows(drawing_labels, stats, skin, parts)
    pictures = []
    for i, letter in enumerate(LETTERS):
        rgba = cut_out(image, drawing_labels, parts[i] | arrows[i])
        cv2.imwrite(str(SIGNS_DIR / f"{letter}.png"), rgba)
        pictures.append((letter, rgba))
    cv2.imwrite(str(SIGNS_DIR / "_preview.png"), preview(pictures))
    print(f"Saved {len(pictures)} pictures to {SIGNS_DIR} (check _preview.png)")


if __name__ == "__main__":
    main()
