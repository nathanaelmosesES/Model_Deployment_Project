import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from credit_scoring.inference import predict_record

def main():
    parser = argparse.ArgumentParser(description="Predict satu nasabah dari file JSON.")
    parser.add_argument("--json", required=True, help="Path payload JSON.")
    parser.add_argument("--model", default="artifacts/model.joblib", help="Path model joblib.")
    args = parser.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    result = predict_record(payload, args.model)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
