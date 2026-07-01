import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from credit_scoring.train import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    train_experiments,
)


def main():
    parser = argparse.ArgumentParser(description="Train credit scoring ML pipeline.")
    parser.add_argument("--data", default="data_C.csv", help="Path ke dataset CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Folder artifact model.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Jumlah fold cross-validation. Default 3 agar training lebih cepat.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI. Default: folder `mlruns` lokal.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help="Nama MLflow experiment.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Matikan progress bar training.",
    )
    args = parser.parse_args()

    result = train_experiments(
        data_path=args.data,
        output_dir=args.output_dir,
        random_state=args.random_state,
        show_progress=not args.no_progress,
        cv_splits=args.cv_splits,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
