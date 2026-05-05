import numpy as np
import pandas as pd

from utils.triple_barrier import encode_triple_barrier_labels, \
    triple_barrier_labels


CNN_FEATURE_COLUMNS = [
    "log_return_1d",
    "close_sma_ratio_5",
    "close_sma_ratio_10",
    "close_sma_ratio_20",
    "close_sma_ratio_50",
    "rsi_14",
    "macd_hist_norm",
    "bbp_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volume_ratio_5",
    "volume_ratio_20",
]


def load_ohlcv_csv(csv_path):
    """
    Load OHLCV data from a CSV file with the following columns:
    Date, Open, High, Low, Close, Volume and return a DataFrame indexed by Date.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("Date")
    df = df.set_index("Date")

    return df


def add_returns(df):
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))
    return out


def add_sma(df, windows=(5, 10, 20, 50)):
    out = df.copy()
    for w in windows:
        out[f"sma_{w}"] = out["Close"].rolling(w).mean()
        out[f"close_sma_ratio_{w}"] = out["Close"] / out[f"sma_{w}"]
    return out


def add_ema(df, spans=(12, 26)):
    out = df.copy()
    for s in spans:
        out[f"ema_{s}"] = out["Close"].ewm(span=s, adjust=False).mean()
    return out


def add_rsi(df, window=14):
    out = df.copy()
    delta = out["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(df, fast=12, slow=26, signal=9):
    out = df.copy()

    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()

    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    return out


def add_bollinger_bands(df, window=20, num_std=2.0):
    out = df.copy()

    mid = out["Close"].rolling(window).mean()
    std = out["Close"].rolling(window).std()

    out[f"bb_mid_{window}"] = mid
    out[f"bb_upper_{window}"] = mid + num_std * std
    out[f"bb_lower_{window}"] = mid - num_std * std
    out[f"bbp_{window}"] = (
        (out["Close"] - out[f"bb_lower_{window}"]) /
        (out[f"bb_upper_{window}"] - out[f"bb_lower_{window}"])
    )

    return out


def add_momentum(df, windows=(5, 10, 20)):
    out = df.copy()
    for w in windows:
        out[f"momentum_{w}"] = out["Close"] / out["Close"].shift(w) - 1.0
    return out


def add_volatility(df, windows=(5, 10, 20)):
    out = df.copy()
    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change()

    for w in windows:
        out[f"volatility_{w}"] = out["return_1d"].rolling(w).std()

    return out


def add_volume_features(df, windows=(5, 20)):
    out = df.copy()
    for w in windows:
        out[f"volume_sma_{w}"] = out["Volume"].rolling(w).mean()
        out[f"volume_ratio_{w}"] = out["Volume"] / out[f"volume_sma_{w}"]
    return out


def compute_all_indicators(df):
    out = df.copy()
    out = add_returns(out)
    out = add_sma(out)
    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_momentum(out)
    out = add_volatility(out)
    out = add_volume_features(out)
    return out


def compute_cnn_features(df):
    out = df.copy()
    out = add_returns(out)
    out = add_sma(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_momentum(out)
    out = add_volatility(out)
    out = add_volume_features(out)

    out["macd_hist_norm"] = out["macd_hist"] / out["Close"]

    return out[CNN_FEATURE_COLUMNS].copy()


def save_indicators_to_csv(input_csv_path, output_csv_path, dropna=False):
    # Load OHLCV data, compute indicators, save to a new CSV.
    # Returns a DataFrame with indicators

    df = load_ohlcv_csv(input_csv_path)
    df = compute_all_indicators(df)

    if dropna:
        df = df.dropna()

    df.to_csv(output_csv_path, index=True)
    return df

def save_cnn_features_to_csv(input_csv_path, output_csv_path, dropna=True):
    df = load_ohlcv_csv(input_csv_path)
    df = compute_cnn_features(df)

    if dropna:
        df = df.dropna()

    df.to_csv(output_csv_path, index=True)
    return df


def compute_cnn_training_data(
    df,
    upper_barrier=0.03,
    lower_barrier=-0.03,
    max_holding_period=10,
    price_col="Close",
    drop_feature_na=True,
    drop_last_incomplete_labels=True,
):
    # Build the aligned CNN feature matrix and triple-barrier labels.

    # Returns a DataFrame indexed by Date with the feature columns plus the label:
    # 0 = Hold
    # 1 = Buy
    # 2 = Sell
    features = compute_cnn_features(df)
    if drop_feature_na:
        features = features.dropna()

    raw_labels = triple_barrier_labels(
        df,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        max_holding_period=max_holding_period,
        price_col=price_col,
        drop_last_incomplete=drop_last_incomplete_labels,
    )
    labels = encode_triple_barrier_labels(raw_labels).rename("label")

    dataset = features.join(labels, how="inner")
    dataset = dataset.dropna(subset=["label"])
    dataset["label"] = dataset["label"].astype(np.int64)
    
    return dataset


def save_cnn_training_data_to_csv(
    input_csv_path,
    output_csv_path,
    upper_barrier=0.03,
    lower_barrier=-0.03,
    max_holding_period=10,
    price_col="Close",
    drop_feature_na=True,
    drop_last_incomplete_labels=True,
):
    # Load the OHLCV data, compute CNN features, align triple-barrier labels, and save
    df = load_ohlcv_csv(input_csv_path)
    dataset = compute_cnn_training_data(
        df,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        max_holding_period=max_holding_period,
        price_col=price_col,
        drop_feature_na=drop_feature_na,
        drop_last_incomplete_labels=drop_last_incomplete_labels,
    )
    dataset.to_csv(output_csv_path, index=True)
    return dataset
