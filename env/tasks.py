import json
import os

def load_task(name: str):
    path = os.path.join("data", f"{name}.json")
    with open(path) as f:
        return json.load(f)