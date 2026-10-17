import json

examples = []

def save_examples():
    with open("examples.json", "w") as f:
        json.dump(examples, f)