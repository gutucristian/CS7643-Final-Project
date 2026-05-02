# to train the lstm, run this:
#       python experiments/train_lstm.py --config configs/lstm.yaml


import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
import shutil
from torch.utils.data import Dataset, DataLoader

# fix path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm.model import LSTMModel
from training.losses import cross_entropy_loss, focal_loss, sharpe_loss
from training.trainer import Trainer


class SequenceDataset(Dataset):
    # prep time series data and get forward looking return

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


# get device/resources
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


def build_run_dirs(project_root: str, run_tag: str):
    experiments_root = os.path.join(project_root, "results", "lstm_experiments")
    run_dir = os.path.join(experiments_root, run_tag)
    return experiments_root, run_dir


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0.0

    for windows, labels, fwd_returns in dataloader:
        windows = windows.to(device)
        labels = labels.to(device)
        fwd_returns = fwd_returns.to(device)

        optimizer.zero_grad()
        logits = model(windows)
        loss = criterion(logits, labels, fwd_returns)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


def predict_prob(model, dataloader, device):
    # heloper functiont to run inferences
    model.eval()
    all_probs = []

    with torch.no_grad():

        for w, l, f in dataloader:
            w = w.to(device)

            # get logits
            logits = model(w)
            
            probs = torch.softmax(
                logits,
                dim =- 1
            ).cpu().numpy()

            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def main():
    # create parse and add args for processing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type = str, 
        default = "configs/lstm_base.yaml"
    )
    args = parser.parse_args()
    run_tag = os.path.splitext(os.path.basename(args.config))[0]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiments_root, run_dir = build_run_dirs(project_root, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    config = load_config(args.config)

    # get data info
    data_info = config["data"]
    csv_path = data_info["csv_path"]
    feature_cols = data_info["feature_cols"]
    label_col = data_info.get("label_col", "label")
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

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    labels = df[label_col].astype(int)
    features_df = df[feature_cols].copy()
    raw_prices = df[price_col].copy()

    # get label summary
    label_pcts = (labels.value_counts(normalize = True).sort_index() * 100).round(2)
    label_counts = labels.value_counts().sort_index()

    print(f"Total rows: {len(df)}")
    print(f"Using label column: {label_col}")
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
        dropout = model_config["dropout"],
        pooling = model_config.get("pooling", "mean"),
        head_hidden_size = model_config.get("head_hidden_size"),
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

        if training_config["loss"] == "focal": # focal loss
            focal_gamma = float(training_config.get("focal_gamma", 2.0))
            
            print(f"Using focal loss with gamma = {focal_gamma}")
            
            criterion = focal_loss(
                class_counts = class_counts,
                num_classes = model_config["num_classes"],
                gamma = focal_gamma,
            )

        else: # default to cross entropy
            criterion = cross_entropy_loss(
                class_counts = class_counts,
                num_classes = model_config["num_classes"]
            )

    trainer = Trainer(model, optimizer, criterion, device)

    os.makedirs(checkpoint_config["dir"], exist_ok = True)

    checkpoint_path = os.path.join(checkpoint_config["dir"], f"{run_tag}_best.pt")

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(1, training_config["epochs"] + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            trainer.device,
            grad_clip = training_config.get("grad_clip"),
        )

        val_metrics = trainer.evaluate(val_loader)

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

    print(f"\nTest results | loss = {test_metrics['loss']:.4f} | acc = {test_metrics['accuracy']:.4f}")

    raw_preds = trainer.predict(test_loader)
    pred_probs = predict_prob(model, test_loader, trainer.device)

    # get pred start and end
    pred_start = window_size - 1
    pred_end = len(y_test) - return_horizon

    # get actual vs pred
    true_labels = y_test.iloc[pred_start:pred_end].values
    pred_dates = df.iloc[test_sl]["Date"].iloc[pred_start:pred_end].values

    if len(raw_preds) != len(true_labels):
        raise ValueError(f"Prediction alignment mismatch: got {len(raw_preds)} preds but {len(true_labels)} labels.")

    res_df_input = {
        "true_label": true_labels,
        "pred_label": raw_preds,
        "prob_hold": pred_probs[:, 0],
        "prob_long": pred_probs[:, 1],
        "prob_short": pred_probs[:, 2],
        "position_score": pred_probs[:, 1] - pred_probs[:, 2]
    }

    results_df = pd.DataFrame(res_df_input, index = pd.to_datetime(pred_dates))


    results_path = os.path.join(run_dir, "test_predictions.csv")
    results_df.to_csv(results_path)
    
    print(f"Saved test labels + predictions to: {results_path}")

    config_copy_path = os.path.join(run_dir, "config.yaml")
    
    shutil.copyfile(args.config, config_copy_path)
    
    print(f"Saved run config to: {config_copy_path}")


if __name__ == "__main__":
    main()





