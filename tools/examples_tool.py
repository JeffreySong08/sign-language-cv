"""
examples_tool.py - look at and clean up the example files in data/.

  python -m tools.examples_tool count
      Shows how many examples each letter has, in total and per file.

  python -m tools.examples_tool remove C jeffrey_left
      Deletes ALL examples of letter C from data/examples_jeffrey_left.json,
      so you can record C again. (Saves a backup copy first.)

Every script loads all data/examples_*.json files automatically, so there
is no merge step: just put a file in data/ (or git pull a teammate's).
"""

import json
import os
import shutil
import sys
from collections import Counter
from signlingo.features import NUM_NUMBERS, load_all_examples
from signlingo.paths import DATA_DIR


def show_counts(examples):
    counts = Counter(e["letter"] for e in examples)
    print("  " + "  ".join(f"{L}:{counts[L]}" for L in sorted(counts)))
    print(f"  total: {len(examples)}")


command = sys.argv[1] if len(sys.argv) > 1 else "count"

if command == "count":
    examples = load_all_examples()
    show_counts(examples)

elif command == "remove" and len(sys.argv) == 4:
    letter, name = sys.argv[2].upper(), sys.argv[3]
    path = os.path.join(DATA_DIR, f"examples_{name}.json")
    with open(path) as f:
        examples = json.load(f)
    shutil.copy(path, path.replace(".json", "_backup.json"))
    kept = [e for e in examples if e["letter"] != letter]
    with open(path, "w") as f:
        json.dump(kept, f)
    print(f"Removed {len(examples) - len(kept)} examples of {letter} from {path}")
    show_counts([e for e in kept if len(e["numbers"]) == NUM_NUMBERS])

else:
    print(__doc__)
