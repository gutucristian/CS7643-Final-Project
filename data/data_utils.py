# utils/data_utils.py

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a standard CSV with columns:
    Date, Open, High, Low, Close, Volume

    Returns a DataFrame indexed by Date.
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


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))
    return out


def add_sma(df: pd.DataFrame, windows=(5, 10, 20, 50)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"sma_{w}"] = out["Close"].rolling(w).mean()
        out[f"close_sma_ratio_{w}"] = out["Close"] / out[f"sma_{w}"]
    return out


def add_ema(df: pd.DataFrame, spans=(12, 26)) -> pd.DataFrame:
    out = df.copy()
    for s in spans:
        out[f"ema_{s}"] = out["Close"].ewm(span=s, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    out = df.copy()

    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()

    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    return out


def add_bollinger_bands(df: pd.DataFrame, window=20, num_std=2.0) -> pd.DataFrame:
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


def add_momentum(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"momentum_{w}"] = out["Close"] / out["Close"].shift(w) - 1.0
    return out


def add_volatility(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    out = df.copy()
    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change()

    for w in windows:
        out[f"volatility_{w}"] = out["return_1d"].rolling(w).std()

    return out


def add_volume_features(df: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"volume_sma_{w}"] = out["Volume"].rolling(w).mean()
        out[f"volume_ratio_{w}"] = out["Volume"] / out[f"volume_sma_{w}"]
    return out


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def save_indicators_to_csv(input_csv_path: str, output_csv_path: str, dropna: bool = False) -> pd.DataFrame:
    """
    Load OHLCV data, compute indicators, and save to a new CSV.

    Args:
        input_csv_path: path to raw OHLCV CSV
        output_csv_path: path to save enriched CSV
        dropna: whether to drop rows with NaNs caused by rolling indicators

    Returns:
        DataFrame with indicators
    """
    df = load_ohlcv_csv(input_csv_path)
    df = compute_all_indicators(df)

    if dropna:
        df = df.dropna()

    df.to_csv(output_csv_path, index=True)
    return df
