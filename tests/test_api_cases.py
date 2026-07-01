import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


CASES = [
    "sample_payload_poor.json",
    "sample_payload_standard.json",
    "sample_payload_good.json",
]


def main():
    parser = argparse.ArgumentParser(description="Smoke test deployed credit API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for case in CASES:
        payload = json.loads((root / case).read_text(encoding="utf-8"))
        response = requests.post(f"{args.base_url}/predict", json=payload, timeout=30)
        response.raise_for_status()
        print(case)
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
