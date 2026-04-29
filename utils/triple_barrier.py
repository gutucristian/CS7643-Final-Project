from __future__ import annotations

from pathlib import Path

import pandas as pd


def _prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input price data to a sorted frame with a Date column.
    """
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


def triple_barrier_labels(
    df,
    upper_barrier,
    lower_barrier,
    max_holding_period,
    price_col,
    drop_last_incomplete: bool = False,
):
    """
    Calculate labels using the triple barrier method.

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

    n = len(data)
    labels = pd.Series(0, index=data["Date"], dtype="int8", name="label")

    for i in range(n):
        entry_price = prices[i]
        upper_price = entry_price * (1 + upper_barrier)
        lower_price = entry_price * (1 + lower_barrier)

        end_idx = min(i + max_holding_period, n - 1)
        label = 0

        for j in range(i + 1, end_idx + 1):

            
            # get upper indication
            hit_upper = False
            if highs[j] >= upper_price:
                hit_upper = True
            
            # get lower indication
            hit_lower = False
            if lows[j] <= lower_price:
                hit_lower = True

            if hit_upper and hit_lower:
                if opens is not None:
                    dist_to_upper = abs(opens[j] - upper_price)
                    dist_to_lower = abs(opens[j] - lower_price)
                    label = 1 if dist_to_upper < dist_to_lower else -1
                else:
                    label = 0
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


def encode_triple_barrier_labels(labels: pd.Series) -> pd.Series:
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
    """
    wraper class for labeling
    """

    def __init__(
        self,
        file_path,
        upper_barrier,
        lower_barrier,
        max_holding_period,
        price_col,
    ):
        self.file_path = file_path
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.max_holding_period = max_holding_period
        self.price_col = price_col

        self.df = pd.read_csv(self.file_path)

    def run(self, encoded: bool = False, drop_last_incomplete: bool = False) -> pd.DataFrame:
        data = _prepare_price_data(self.df).set_index("Date")
        labels = triple_barrier_labels(
            data,
            upper_barrier=self.upper_barrier,
            lower_barrier=self.lower_barrier,
            max_holding_period=self.max_holding_period,
            price_col=self.price_col,
            drop_last_incomplete=drop_last_incomplete,
        )

        if encoded:
            labels = encode_triple_barrier_labels(labels)

        data["label"] = labels
        if drop_last_incomplete:
            data = data.loc[labels.index]

        return data.reset_index()


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "SPY_ohlcv_with_indicators.csv"
    output_path = project_root / "SPY_with_labels_triple_barrier.csv"

    labeler = TripleBarrierLabeler(
        # may need to tune these hyper parameters for labeling strategies. 
        # can include this hyper paramater tuning as part of the model training experiments
        # current distribution of lables. may wanrt to use focal loass to handle class imbalance
            # 0    2561
            # -1    1224
            # 1    1199
        file_path=input_path,
        upper_barrier=0.03,
        lower_barrier=-0.03,
        max_holding_period=10,
        price_col="Close",
    )

    df_labeled = labeler.run(drop_last_incomplete=True)

    df_labeled.to_csv(output_path, index=False)

    print("Labeling complete.")
    print(df_labeled["label"].value_counts())


if __name__ == "__main__":
    main()
