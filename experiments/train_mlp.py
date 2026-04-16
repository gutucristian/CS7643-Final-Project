"""
End-to-end MLP training and backtesting script.

Usage:
    python experiments/train_mlp.py --config configs/mlp.yaml
"""

import argparse
import os
import sys

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.data_utils import load_ohlcv_csv
from data.dataset import SPYDataset
from data.labeling import direction_change_labels
from models.mlp.model import MLP
from training.losses import cross_entropy_loss, sharpe_loss
from training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def chronological_split(n: int, train_frac: float, val_frac: float):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def load_or_generate_labels(df, csv_path: str, price_col: str) -> "pd.Series":
    label_path = os.path.splitext(csv_path)[0] + "_labels_mlp.csv"
    if os.path.exists(label_path):
        print(f"Loading cached labels from {label_path}")
        labels = pd.read_csv(label_path, index_col=0, parse_dates=True).squeeze()
        labels.index = pd.to_datetime(labels.index)
        labels = labels.astype(int)
    else:
        print(f"Generating labels and saving to {label_path}")
        labels = direction_change_labels(df, price_col=price_col)
        labels.to_csv(label_path, header=["label"])
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp.yaml")
    args = parser.parse_args()

    # derive run tag from config filename, e.g. configs/mlp_lr001.yaml → mlp_lr001
    run_tag = os.path.splitext(os.path.basename(args.config))[0]

    cfg = load_config(args.config)
    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]
    bc = cfg["backtest"]
    cc = cfg["checkpoint"]

    # ------------------------------------------------------------------ data
    df = load_ohlcv_csv(dc["csv_path"])
    labels = load_or_generate_labels(df, dc["csv_path"], dc["price_col"])

    # align df to label index
    df = df.loc[labels.index]

    n = len(df)
    train_sl, val_sl, test_sl = chronological_split(n, dc["train_frac"], dc["val_frac"])

    df_train = df.iloc[train_sl]
    df_val   = df.iloc[val_sl]
    df_test  = df.iloc[test_sl]
    lbl_train = labels.iloc[train_sl]
    lbl_val   = labels.iloc[val_sl]
    lbl_test  = labels.iloc[test_sl]

    print(f"Data split — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    # convert OHLC to log-returns and volume to log-diff so features are stationary
    feature_cols = dc["feature_cols"]
    price_cols  = [c for c in feature_cols if c != "Volume"]
    has_volume  = "Volume" in feature_cols

    def make_stationary(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        for col in price_cols:
            df_out[col] = np.log(df_in[col] / df_in[col].shift(1))
        if has_volume:
            df_out["Volume"] = np.log(df_in["Volume"] / df_in["Volume"].shift(1))
        return df_out.iloc[1:]   # drop first row (NaN from shift)

    df_train = make_stationary(df_train)
    df_val   = make_stationary(df_val)
    df_test  = make_stationary(df_test)

    # re-align labels after dropping first row of each split
    lbl_train = lbl_train.loc[lbl_train.index.intersection(df_train.index)]
    lbl_val   = lbl_val.loc[lbl_val.index.intersection(df_val.index)]
    lbl_test  = lbl_test.loc[lbl_test.index.intersection(df_test.index)]

    # z-score normalise using train statistics only
    means = df_train[feature_cols].mean()
    stds  = df_train[feature_cols].std().replace(0, 1)

    df_train[feature_cols] = (df_train[feature_cols] - means) / stds
    df_val[feature_cols]   = (df_val[feature_cols]   - means) / stds
    df_test[feature_cols]  = (df_test[feature_cols]  - means) / stds

    window_size = dc["window_size"]
    train_ds = SPYDataset(df_train, lbl_train, window_size, feature_cols, dc["price_col"])
    val_ds   = SPYDataset(df_val,   lbl_val,   window_size, feature_cols, dc["price_col"])
    test_ds  = SPYDataset(df_test,  lbl_test,  window_size, feature_cols, dc["price_col"])

    batch_size = tc["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # ----------------------------------------------------------------- model
    input_size = window_size * len(feature_cols)
    model = MLP(
        input_size=input_size,
        hidden_sizes=mc["hidden_sizes"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
    )
    print(f"Model: MLP  input={input_size}  hidden={mc['hidden_sizes']}  classes={mc['num_classes']}")

    device = resolve_device(tc["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])

    if tc["loss"] == "sharpe":
        criterion = sharpe_loss()
    else:
        class_counts = [int((lbl_train == c).sum()) for c in range(mc["num_classes"])]
        print(f"Train label counts — Hold: {class_counts[0]}, Long: {class_counts[1]}, Short: {class_counts[2]}")
        criterion = cross_entropy_loss(class_counts=class_counts, num_classes=mc["num_classes"])
    print(f"Loss: {tc['loss']}  |  Device: {device}")

    trainer = Trainer(model, optimizer, criterion, device)

    # --------------------------------------------------------------- training
    os.makedirs(cc["dir"], exist_ok=True)
    checkpoint_path = os.path.join(cc["dir"], f"{run_tag}_best.pt")

    best_val_loss = float("inf")
    print()
    for epoch in range(1, tc["epochs"] + 1):
        # single-epoch train
        history = trainer.train(train_loader, epochs=1)
        val_metrics = trainer.evaluate(val_loader)

        train_loss = history["train_loss"][0]
        val_loss   = val_metrics["loss"]
        val_acc    = val_metrics["accuracy"]

        print(
            f"Epoch {epoch:3d}/{tc['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                checkpoint_path,
            )

    print(f"\nBest val loss: {best_val_loss:.4f}  |  checkpoint: {checkpoint_path}")

    # --------------------------------------------------- reload best & test
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest  loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.4f}")

    # class breakdown
    raw_preds = trainer.predict(test_loader)
    unique, counts = np.unique(raw_preds, return_counts=True)
    label_names = {0: "Hold", 1: "Long", 2: "Short"}
    print("Prediction distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")

    # ------------------------------------------------- save predictions
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp")
    os.makedirs(out_dir, exist_ok=True)

    # true labels aligned with predictions: last label of each window
    true_labels = lbl_test.values[window_size - 1:]
    pred_index  = lbl_test.index[window_size - 1:]

    results_df = pd.DataFrame(
        {"true_label": true_labels, "pred_label": raw_preds},
        index=pred_index,
    )
    results_path = os.path.join(out_dir, f"{run_tag}_test_predictions.csv")
    results_df.to_csv(results_path)
    print(f"\nSaved test labels + predictions → {results_path}")

    print(f"\nRun experiments/backtest_mlp.py --config {args.config} to evaluate backtest performance.")


if __name__ == "__main__":
    main()
