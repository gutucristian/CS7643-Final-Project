import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_utils import CNN_FEATURE_COLUMNS, save_cnn_features_to_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the features used for the CNN model."
    )
    parser.add_argument(
        "--input",
        default="SPY_ohlcv.csv",
        help="Path to the raw OHLCV CSV",
    )
    parser.add_argument(
        "--output",
        default="SPY_cnn_features.csv",
        help="Path to save the CNN feature CSV",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep warm-up rows with NAN instead of dropping",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_cnn_features_to_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        dropna=not args.keep_na,
    )


if __name__ == "__main__":
    main()
