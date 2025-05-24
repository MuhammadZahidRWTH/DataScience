import json
import os

def load_extraction_schema(path="field_definitions/extraction-fields.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[✘] Schema file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
