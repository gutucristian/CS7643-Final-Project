import yfinance as yf
import pandas as pd

def download_etf_data(ticker: str, start: str, end: str, interval: str = "1d", output_file: str = None) -> pd.DataFrame:
    """
    Download OHLCV data for an ETF from Yahoo Finance.

    Args:
        ticker: ETF ticker symbol, e.g. "SPY"
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        interval: Price data interval
        output_file: Optional CSV file path to save results

    Returns:
        pandas DataFrame with Open, High, Low, Close, Volume
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Reset index so Date becomes a normal column
    df.reset_index(inplace=True)

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved data to {output_file}")

    return df


if __name__ == "__main__":
    ticker = "SPY"
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    etf_data = download_etf_data(
        ticker=ticker,
        start=start_date,
        end=end_date,
        output_file=f"{ticker}_ohlcv.csv"
    )

    print(etf_data.head())