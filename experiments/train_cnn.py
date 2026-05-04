# End-to-end training for the CNN model. To run use: python experiments/train_cnn.py --config configs/cnn.yaml

import argparse
import os
import sys

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data.dataset import SPYDataset
from models.cnn.model import CNN1D
from training.losses import cross_entropy_loss, focal_loss, sharpe_loss
from training.trainer import Trainer, resolve_device


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def chronological_split(n, train_frac, val_frac):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def load_training_frame(csv_path, label_col):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    labels = df[label_col].astype(int)
    features = df.drop(columns=[label_col])
    return features, labels


def build_forward_returns(series, kind="simple"):
    shifted = series.shift(-1)
    if kind == "log":
        return np.expm1(shifted).fillna(0.0)
    if kind == "simple":
        return shifted.fillna(0.0)
    raise ValueError(f"Unsupported fwd_return_kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()

    run_tag = os.path.splitext(os.path.basename(args.config))[0]

    cfg = load_config(args.config)
    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]
    cc = cfg["checkpoint"]

    df, labels = load_training_frame(dc["csv_path"], dc["label_col"])

    feature_cols = dc["feature_cols"]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing configured feature columns: {missing_cols}")

    fwd_return_col = dc.get("fwd_return_col")
    fwd_return_kind = dc.get("fwd_return_kind", "simple")
    if fwd_return_col is None:
        raise ValueError("cnn config must define data.fwd_return_col")
    if fwd_return_col not in df.columns:
        raise ValueError(f"fwd_return_col '{fwd_return_col}' not found in dataset columns")

    fwd_returns = build_forward_returns(df[fwd_return_col], kind=fwd_return_kind)

    n = len(df)
    train_sl, val_sl, test_sl = chronological_split(n, dc["train_frac"], dc["val_frac"])

    df_train = df.iloc[train_sl].copy()
    df_val = df.iloc[val_sl].copy()
    df_test = df.iloc[test_sl].copy()

    lbl_train = labels.iloc[train_sl].copy()
    lbl_val = labels.iloc[val_sl].copy()
    lbl_test = labels.iloc[test_sl].copy()

    fwd_train = fwd_returns.iloc[train_sl].copy()
    fwd_val = fwd_returns.iloc[val_sl].copy()
    fwd_test = fwd_returns.iloc[test_sl].copy()

    print(f"Data split — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1)

    df_train.loc[:, feature_cols] = (df_train[feature_cols] - means) / stds
    df_val.loc[:, feature_cols] = (df_val[feature_cols] - means) / stds
    df_test.loc[:, feature_cols] = (df_test[feature_cols] - means) / stds

    window_size = dc["window_size"]
    train_ds = SPYDataset(
        df_train,
        lbl_train,
        window_size=window_size,
        feature_cols=feature_cols,
        fwd_returns=fwd_train,
    )
    val_ds = SPYDataset(
        df_val,
        lbl_val,
        window_size=window_size,
        feature_cols=feature_cols,
        fwd_returns=fwd_val,
    )
    test_ds = SPYDataset(
        df_test,
        lbl_test,
        window_size=window_size,
        feature_cols=feature_cols,
        fwd_returns=fwd_test,
    )

    batch_size = tc["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    first_windows, first_labels, _ = next(iter(train_loader))
    print(
        f"Sample batch — windows: {tuple(first_windows.shape)}, "
        f"labels: {tuple(first_labels.shape)}"
    )

    model = CNN1D(
        input_channels=len(feature_cols),
        conv_channels=mc["conv_channels"],
        kernel_size=mc["kernel_size"],
        hidden_dim=mc["hidden_dim"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
    )
    print(
        f"Model: CNN1D  channels={mc['conv_channels']}  kernel={mc['kernel_size']}  "
        f"hidden={mc['hidden_dim']}  classes={mc['num_classes']}"
    )

    device = resolve_device(tc["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])

    if tc["loss"] == "focal":
        print(f"Focal gamma: {tc.get('focal_gamma', 2.0)}")
        if tc.get("focal_alpha") is not None:
            print(f"Focal alpha: {tc['focal_alpha']}")
        criterion = focal_loss(
            num_classes=mc["num_classes"],
            gamma=tc.get("focal_gamma", 2.0),
            alpha=tc.get("focal_alpha"),
        )
    elif tc["loss"] == "sharpe":
        criterion = sharpe_loss()
    else:
        class_counts = [int((lbl_train == c).sum()) for c in range(mc["num_classes"])]
        print(f"Train label counts — Hold: {class_counts[0]}, Long: {class_counts[1]}, Short: {class_counts[2]}")
        if tc.get("class_weight_multipliers") is not None:
            print(f"Class weight multipliers: {tc['class_weight_multipliers']}")
        criterion = cross_entropy_loss(
            class_counts=class_counts,
            num_classes=mc["num_classes"],
            class_weight_multipliers=tc.get("class_weight_multipliers"),
        )
    print(f"Loss: {tc['loss']}  |  Device: {device}")

    trainer = Trainer(model, optimizer, criterion, device)

    os.makedirs(cc["dir"], exist_ok=True)
    checkpoint_path = os.path.join(cc["dir"], f"{run_tag}_best.pt")

    best_val_loss = float("inf")
    print()
    for epoch in range(1, tc["epochs"] + 1):
        history = trainer.train(train_loader, epochs=1)
        val_metrics = trainer.evaluate(val_loader)

        train_loss = history["train_loss"][0]
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest  loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.4f}")

    raw_preds = trainer.predict(test_loader)
    unique, counts = np.unique(raw_preds, return_counts=True)
    label_names = {0: "Hold", 1: "Long", 2: "Short"}
    print("Prediction distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn")
    os.makedirs(out_dir, exist_ok=True)

    true_labels = lbl_test.values[window_size - 1:]
    pred_index = lbl_test.index[window_size - 1:]

    cm = confusion_matrix(true_labels, raw_preds, labels=[0, 1, 2])
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("          Hold  Long Short")
    for idx, row in enumerate(cm):
        print(f"{label_names[idx]:>5} {row}")

    report = classification_report(
        true_labels,
        raw_preds,
        labels=[0, 1, 2],
        target_names=["Hold", "Long", "Short"],
        digits=4,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report)

    results_df = pd.DataFrame(
        {"true_label": true_labels, "pred_label": raw_preds},
        index=pred_index,
    )
    results_path = os.path.join(out_dir, f"{run_tag}_test_predictions.csv")
    results_df.to_csv(results_path)
    print(f"\nSaved test labels + predictions -> {results_path}")
    print(f"\nRun experiments/backtest_cnn.py --config {args.config} to evaluate backtest performance.")


if __name__ == "__main__":
    main()
