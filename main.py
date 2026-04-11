from utils.data_utils import save_indicators_to_csv

def main():
    input_csv = "SPY_ohlcv.csv"
    output_csv = "SPY_ohlcv_with_indicators.csv"

    df = save_indicators_to_csv(input_csv, output_csv, dropna=True)
    print(df.head())
    print(f"Saved file to {output_csv}")

if __name__ == "__main__":
    main()