from pathlib import Path
import pandas as pd

def _prepare_price_data(df):
    # Normalize input price data to a sorted frame with a Date column
    data = df.copy()

    if "Date" not in data.columns:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Expected a Date column or a DatetimeIndex.")
        data = data.reset_index()
        if "Date" not in data.columns:
            data = data.rename(columns={data.columns[0]: "Date"})

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def build_label_col_name(
    barrier_mode,
    max_holding_period,
    upper_barrier=None,
    lower_barrier=None,
    vol_window=None,
    vol_multiplier=None,
):
    if barrier_mode == "fixed":
        upper_pct = int(round(float(upper_barrier) * 100))
        lower_pct = int(round(abs(float(lower_barrier)) * 100))
        return f"label_tb_fixed_up{upper_pct}_down{lower_pct}_{max_holding_period}d"

    if barrier_mode == "volatility":
        mult_str = str(vol_multiplier).replace(".", "p")
        return f"label_tb_vol_w{vol_window}_k{mult_str}_{max_holding_period}d"

    raise ValueError(f"Unsupported barrier_mode: {barrier_mode}")


def _rolling_volatility(df, price_col, vol_window):
    returns = df[price_col].pct_change()
    return returns.rolling(vol_window).std()


def triple_barrier_labels(
    df,
    upper_barrier,
    lower_barrier,
    max_holding_period,
    price_col,
    barrier_mode="fixed",
    vol_window=None,
    vol_multiplier=None,
    drop_last_incomplete=False,
):
    """
    Calculate labels using the triple barrier method

    Returns labels indexed by Date using the raw encoding:
      1 = upper barrier hit first
      0 = no barrier hit within the horizon
     -1 = lower barrier hit first
    """
    data = _prepare_price_data(df)

    prices = data[price_col].values
    highs = data["High"].values
    lows = data["Low"].values
    opens = data["Open"].values
    rolling_vol = None

    if barrier_mode == "volatility":
        rolling_vol = _rolling_volatility(
            data,
            price_col=price_col,
            vol_window=vol_window,
        ).values

    n = len(data)
    labels = pd.Series(0, index=data["Date"], dtype="int8", name="label")

    for i in range(n):
        entry_price = prices[i]

        if barrier_mode == "fixed":
            upper_width = float(upper_barrier)
            lower_width = abs(float(lower_barrier))
        elif barrier_mode == "volatility":
            vol_i = rolling_vol[i]
            if pd.isna(vol_i):
                labels.iloc[i] = 0
                continue
            upper_width = float(vol_multiplier) * float(vol_i)
            lower_width = upper_width
        else:
            raise ValueError(f"Unsupported barrier_mode: {barrier_mode}")

        upper_price = entry_price * (1 + upper_width)
        lower_price = entry_price * (1 - lower_width)

        end_idx = min(i + max_holding_period, n - 1)
        label = 0

        for j in range(i + 1, end_idx + 1):
            hit_upper = highs[j] >= upper_price
            hit_lower = lows[j] <= lower_price

            if hit_upper and hit_lower:
                dist_to_upper = abs(opens[j] - upper_price)
                dist_to_lower = abs(opens[j] - lower_price)
                label = 1 if dist_to_upper < dist_to_lower else -1
                break

            if hit_upper:
                label = 1
                break

            if hit_lower:
                label = -1
                break

        labels.iloc[i] = label

    labels.index.name = "Date"

    if drop_last_incomplete and max_holding_period > 0:
        labels = labels.iloc[:-max_holding_period]

    return labels


def encode_triple_barrier_labels(labels):
    """
    Map raw triple-barrier labels {-1, 0, 1} to {2, 0, 1} for classifiers.
    """
    unexpected = sorted(set(labels.dropna().unique()) - {-1, 0, 1})
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")

    encoded = labels.replace({-1: 2, 0: 0, 1: 1}).astype("int8")
    encoded.name = labels.name or "label"
    return encoded


class TripleBarrierLabeler:
    # wrapper class for labeling

    def __init__(self, file_path, price_col, strategies):
        self.file_path = file_path
        self.price_col = price_col
        self.strategies = strategies

        self.df = pd.read_csv(self.file_path)

    def run(self, encoded=True, drop_last_incomplete=False):
        data = _prepare_price_data(self.df)
        data = data.set_index("Date")
        valid_index = data.index

        for strategy in self.strategies:
            strategy_cfg = strategy.copy()
            label_col = strategy_cfg.pop("label_col", None)
            if label_col is None:
                label_col = build_label_col_name(**strategy_cfg)

            labels = triple_barrier_labels(
                data,
                price_col=self.price_col,
                drop_last_incomplete=drop_last_incomplete,
                **strategy_cfg,
            )

            if encoded:
                labels = encode_triple_barrier_labels(labels)

            data.loc[labels.index, label_col] = labels
            valid_index = valid_index.intersection(labels.index)

        if drop_last_incomplete:
            data = data.loc[valid_index]

        return data.reset_index()


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "SPY_ohlcv_with_indicators.csv"
    output_path = project_root / "SPY_with_labels_triple_barrier.csv"

    strategies = [
        {
            "barrier_mode": "fixed",
            "upper_barrier": 0.03,
            "lower_barrier": -0.03,
            "max_holding_period": 10,
            "label_col": "label",
        },
        {
            "barrier_mode": "fixed",
            "upper_barrier": 0.02,
            "lower_barrier": -0.02,
            "max_holding_period": 10,
        },
        {
            "barrier_mode": "volatility",
            "vol_window": 20,
            "vol_multiplier": 2.0,
            "max_holding_period": 10,
        },
    ]

    labeler = TripleBarrierLabeler(
        file_path=input_path,
        price_col="Close",
        strategies=strategies,
    )

    df_labeled = labeler.run(drop_last_incomplete=True)
    df_labeled.to_csv(output_path, index=False)

    print("Labeling complete.")
    for strategy in strategies:
        strategy_cfg = strategy.copy()
        label_col = strategy_cfg.pop("label_col", None)
        if label_col is None:
            label_col = build_label_col_name(**strategy_cfg)

        print(f"\n{label_col}")
        print(df_labeled[label_col].value_counts())


if __name__ == "__main__":
    main()
