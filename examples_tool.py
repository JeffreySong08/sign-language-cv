"""
examples_tool.py - look at, combine and clean up examples.json.

  python examples_tool.py count
      Shows how many examples each letter has.

  python examples_tool.py merge sara_examples.json kim_examples.json
      Adds the examples from those files into examples.json.
      (Put your teammates' files in this folder first. Give each one a
      different name so they don't overwrite yours.)

  python examples_tool.py remove C
      Deletes ALL examples of letter C, so you can record it again.

Before changing anything, it saves a copy called examples_backup.json,
so you can always go back.
"""

import json
import os
import shutil
import sys
from collections import Counter

EXAMPLES_FILE = "examples.json"
BACKUP_FILE = "examples_backup.json"
NUM_NUMBERS = 63  # must match features.py


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save(examples):
    if os.path.exists(EXAMPLES_FILE):
        shutil.copy(EXAMPLES_FILE, BACKUP_FILE)
    with open(EXAMPLES_FILE, "w") as f:
        json.dump(examples, f)


def show_counts(examples):
    counts = Counter(e["letter"] for e in examples)
    for letter in sorted(counts):
        print(f"  {letter}: {counts[letter]}")
    print(f"  total: {len(examples)}")


examples = load(EXAMPLES_FILE)
command = sys.argv[1] if len(sys.argv) > 1 else "count"

if command == "count":
    show_counts(examples)

elif command == "merge":
    for path in sys.argv[2:]:
        added = skipped = 0
        for e in load(path):
            # Only keep examples made the same way as ours (letter + 63 numbers)
            if not isinstance(e, dict) or "letter" not in e or len(e.get("numbers", [])) != NUM_NUMBERS:
                skipped += 1
            elif e in examples:          # exact same example already there
                skipped += 1
            else:
                examples.append(e)
                added += 1
        print(f"{path}: added {added}, skipped {skipped}")
    save(examples)
    show_counts(examples)

elif command == "remove":
    letter = sys.argv[2].upper()
    before = len(examples)
    examples = [e for e in examples if e["letter"] != letter]
    save(examples)
    print(f"Removed {before - len(examples)} examples of {letter}")
    show_counts(examples)

else:
    print(__doc__)
