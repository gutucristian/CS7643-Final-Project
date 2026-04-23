import pandas as pd


def triple_barrier_labels(
    df,
    upper_barrier,
    lower_barrier,
    max_holding_period,
    price_col,
):
    """
    Calculate labels using the triple brrier method
    """
    
    # copy df
    data = df.copy()

    # convert date to timestamp and order
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop = True)
        

    prices = data[price_col].values

    highs = data["High"].values 
    lows = data["Low"].values 
    opens = data["Open"].values

    n = len(data)
    labels = pd.Series(0, index = data.index, dtype = "int8")

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

    labels.index = data.index

    # remap labels for CrossEntropyLoss:
    # Hold = 0, Buy = 1, Sell = 2
    label_map = {-1: 2, 0: 0, 1: 1}
    labels = labels.map(label_map).astype("int8")
    
    return labels


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

    def run(self) -> pd.DataFrame:
        
        self.df["label"] = triple_barrier_labels(
            self.df,
            upper_barrier = self.upper_barrier,
            lower_barrier = self.lower_barrier,
            max_holding_period = self.max_holding_period,
            price_col = self.price_col,
        )

        return self.df


def main():
    input_path = "../SPY_ohlcv_with_indicators.csv"
    output_path = "../SPY_with_labels_triple_barrier.csv"

    labeler = TripleBarrierLabeler(
        # may need to tune these hyper parameters for labeling strategies. 
        # can include this hyper paramater tuning as part of the model training experiments
        # current distribution of lables. may wanrt to use focal loass to handle class imbalance
            # 0    2561
            # -1    1224
            # 1    1199
        file_path = input_path,
        upper_barrier = 0.03,
        lower_barrier = -0.03,
        max_holding_period = 10,
        price_col = "Close",
    )

    df_labeled = labeler.run()

    df_labeled.to_csv(output_path, index = False)

    print("Labeling complete.")
    print(df_labeled["label"].value_counts())


if __name__ == "__main__":
    main()