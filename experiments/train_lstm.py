# to train the lstm, run this:
#       python experiments/train_lstm.py --config configs/lstm.yaml


import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

# fix path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm.model import LSTMModel
from training.losses import cross_entropy_loss, sharpe_loss
from training.trainer import Trainer


class SequenceDataset(Dataset):
    #prep time series data and get forward looking return

    def __init__(
        self, df_features, labels,
        raw_prices, feature_cols,
        window_size, return_horizon,
    ):
        self.X = df_features[feature_cols].values.astype(np.float32)
        self.y = labels.values.astype(np.int64)

        self.prices = raw_prices.values.astype(np.float32)
        
        self.window_size = window_size
        
        self.return_horizon = return_horizon

    
    def __len__(self):
        
        len_val = len(self.X) - self.window_size - self.return_horizon + 1

        return len_val

    def __getitem__(self, idx):
        idx_window = idx + self.window_size
        x_seq = self.X[idx : idx_window]

        label_idx = idx_window - 1
        y_label = self.y[label_idx]

        entry_price = self.prices[label_idx]
        future_price = self.prices[label_idx + self.return_horizon]

        forward_return = (future_price - entry_price) / entry_price

        items = (torch.tensor(x_seq, dtype = torch.float32), torch.tensor(y_label, dtype = torch.long), torch.tensor(forward_return, dtype = torch.float32))

        return items


def load_config(path):

    with open(path, 'r') as f:

        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":

        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    return device_str


def split(n, train_frac, val_frac):

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    slice1, slice2, slice3 = slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


    return slice1, slice2, slice3


def main():
    # create parse and add args for processing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type = str, 
        default = "configs/lst_base.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # get data info
    data_info = config["data"]
    csv_path = data_info["csv_path"]
    feature_cols = data_info["feature_cols"]
    window_size = data_info["window_size"]
    train_frac = data_info["train_frac"]
    val_frac = data_info["val_frac"]
    price_col = data_info.get("price_col", "Close")
    return_horizon = data_info.get("return_horizon", 10)


    model_config = config["model"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop = True)

    labels = df["label"].astype(int)
    features_df = df[feature_cols].copy()
    raw_prices = df[price_col].copy()

    # get label summary
    label_pcts = (labels.value_counts(normalize = True).sort_index() * 100).round(2)
    label_counts = labels.value_counts().sort_index()

    print(f"Total rows: {len(df)}")
    print("\nLabel counts:")
    print(label_counts)
    print("\nLabel percentages:")
    print(label_pcts)

    n = len(df)
    train_sl, val_sl, test_sl = split(n, train_frac, val_frac)

    X_train = features_df.iloc[train_sl].copy()
    X_val = features_df.iloc[val_sl].copy()
    X_test = features_df.iloc[test_sl].copy()

    y_train = labels.iloc[train_sl].copy()
    y_val = labels.iloc[val_sl].copy()
    y_test = labels.iloc[test_sl].copy()

    raw_train_prices = raw_prices.iloc[train_sl].copy()
    raw_val_prices = raw_prices.iloc[val_sl].copy()
    raw_test_prices = raw_prices.iloc[test_sl].copy()

    print(f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # normalize using train statistics only
    train_mean = X_train.mean()
    train_std = X_train.std().replace(0, 1)

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    train_ds = SequenceDataset(
        X_train, y_train, raw_train_prices,
        feature_cols, window_size, return_horizon
    )

    val_ds = SequenceDataset(
        X_val, y_val, raw_val_prices,
        feature_cols, window_size, return_horizon
    )
    test_ds = SequenceDataset(
        X_test, y_test, raw_test_prices,
        feature_cols, window_size, return_horizon
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = training_config["batch_size"],
        shuffle = True,
        drop_last = False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = training_config["batch_size"],
        shuffle = False,
        drop_last = False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size = training_config["batch_size"],
        shuffle = False,
        drop_last = False
    )

    model = LSTMModel(
        input_size = len(feature_cols),
        hidden_size = model_config["hidden_size"],
        num_layers = model_config["num_layers"],
        num_classes = model_config["num_classes"],
        dropout = model_config["dropout"]
    )

    device = resolve_device(training_config["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = training_config["learning_rate"],
        weight_decay = training_config.get("weight_decay", 0.0),
    )

    if training_config["loss"] == "sharpe":
        criterion = sharpe_loss()
    else:
        class_counts = [int((y_train == c).sum()) for c in range(model_config["num_classes"])]
        print(f"\nTraining label counts -> Hold: {class_counts[0]}, Buy: {class_counts[1]}, Sell: {class_counts[2]}")

        criterion = cross_entropy_loss(
            class_counts = class_counts,
            num_classes = model_config["num_classes"]
        )

    trainer = Trainer(model, optimizer, criterion, device)

    os.makedirs(checkpoint_config["dir"], exist_ok = True)

    checkpoint_path = os.path.join(checkpoint_config["dir"], "lstm_best.pt")

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(1, training_config["epochs"] + 1):

        history = trainer.train(train_loader, epochs=1)
        val_metrics = trainer.evaluate(val_loader)

        train_loss = history["train_loss"][0]
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        log_str = f"Epoch {epoch:3d}/{training_config['epochs']} | "\
                    + f"train_loss={train_loss:.4f} | "\
                    + f"val_loss={val_loss:.4f} | "\
                    +  f"val_acc={val_acc:.4f}"
        
        print(log_str)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_res_summary = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }

            torch.save(model_res_summary, checkpoint_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved best checkpoint to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location = device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTest results | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
