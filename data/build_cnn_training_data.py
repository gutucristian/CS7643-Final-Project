import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_utils import CNN_FEATURE_COLUMNS, \
    save_cnn_training_data_to_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the CNN training dataset with indicator features and triple-barrier labels"
    )
    parser.add_argument(
        "--input",
        default="SPY_ohlcv.csv",
        help="Path to the raw OHLCV CSV",
    )
    parser.add_argument(
        "--output",
        default="SPY_cnn_training_data.csv",
        help="Path to save the aligned CNN training CSV to",
    )
    parser.add_argument(
        "--upper-barrier",
        type=float,
        default=0.03,
        help="Upper barrier for profit taking",
    )
    parser.add_argument(
        "--lower-barrier",
        type=float,
        default=-0.03,
        help="Lower barrier for stop loss",
    )
    parser.add_argument(
        "--holding-period",
        type=int,
        default=10,
        help="Maximum holding period",
    )
    parser.add_argument(
        "--price-col",
        default="Close",
        help="Price column used as the entry price for labeling.",
    )
    parser.add_argument(
        "--keep-feature-na",
        action="store_true",
        help="Keep warm-up rows with NAN instead of dropping",
    )
    parser.add_argument(
        "--keep-truncated-labels",
        action="store_true",
        help="Keep the last rows that do not have the full labeling horizon available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = save_cnn_training_data_to_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        upper_barrier=args.upper_barrier,
        lower_barrier=args.lower_barrier,
        max_holding_period=args.holding_period,
        price_col=args.price_col,
        drop_feature_na=not args.keep_feature_na,
        drop_last_incomplete_labels=not args.keep_truncated_labels,
    )

    print(f"Saved CNN training data to {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Feature columns ({len(CNN_FEATURE_COLUMNS)}): {', '.join(CNN_FEATURE_COLUMNS)}")
    print("Label counts (0=Hold, 1=Buy, 2=Sell):")
    print(df["label"].value_counts().sort_index())
    print(df.head())


if __name__ == "__main__":
    main()
