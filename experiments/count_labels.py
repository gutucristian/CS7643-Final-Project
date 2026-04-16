"""
Count label distribution in a saved labels CSV.

Usage:
    python experiments/count_labels.py SPY_ohlcv_labels_mlp.csv
"""

import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "SPY_ohlcv_labels_mlp.csv"

labels = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()

label_names = {0: "Hold", 1: "Long", 2: "Short"}
total = len(labels)

print(f"File: {path}")
print(f"Total: {total}\n")

for val, name in label_names.items():
    count = (labels == val).sum()
    print(f"  {name} ({val}): {count:>5}  ({count/total*100:.1f}%)")
