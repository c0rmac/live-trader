import numpy as np
import pandas as pd
from signals import datapre

def load_file(file_path: str) -> pd.DataFrame:
    print(file_path)
    file = pd.read_pickle(file_path)

    file['local_time'] = pd.to_datetime(file['local_time'])
    file.reset_index(drop=True, inplace=True)
    file.set_index('timestamp', inplace=True)
    file.sort_index(inplace=True)

    t = pd.Timedelta(hours=24 * 2)
    r = pd.Timedelta(minutes=5)

    def compute_future_values(x):
        future_data = file[(file["local_time"] <= (x + t)) & (file["local_time"] > x + r)]

        if future_data.empty:
            return np.nan, np.nan, np.nan  # Handle empty cases

        future_max = future_data["priceUSD"].max()
        future_min = future_data["priceUSD"].min()
        future_close = future_data["priceUSD"].iloc[-1]
        future_avg = future_data["priceUSD"].mean()
        #closing_log_return = future_data["log_returns"].iloc[-1] if "log_returns" in future_data else np.nan

        return future_max, future_min, future_close, future_avg

    # Apply function once instead of multiple searches
    #file[['future_price_max', 'future_price_min', 'future_price_close', 'future_price_avg']] = file['local_time'].apply(
    #    lambda x: pd.Series(compute_future_values(x))
    #)

    # Compute log returns in vectorized form after extraction
    #file['log_returns'] = np.log(file['future_price_max'] / file["priceUSD"])
    #file['log_returns_min'] = np.log(file['future_price_min'] / file["priceUSD"])
    #file['log_returns_close'] = np.log(file['future_price_close'] / file["priceUSD"])
    #file['log_returns_avg'] = np.log(file['future_price_avg'] / file["priceUSD"])

    return file

def generate_candles(df: pd.DataFrame, interval: str = '1h', max_pains = []):
    """
    Converts price data into candlestick format (OHLC) and includes mode of 'fearGreedIndex' + total 'volume'.
    Ensures NaN values in OHLC data are interpolated, and 'local_time' is set to the last value in each interval.

    Args:
        df (pd.DataFrame): DataFrame containing 'local_time', 'priceUSD', 'fearGreedIndex', and 'volume'.
        interval (str): Resampling interval (e.g., '15T' for 15-minute candles, '1H' for hourly).

    Returns:
        pd.DataFrame: Candlestick data with OHLC, FearGreedIndex mode, volume, and correct timestamp.
    """
    df = df.sort_index()

    # Resample price data into candlestick format
    candle_df = df.resample(interval, on='local_time')['priceUSD'].agg(['first', 'max', 'min', 'last'])

    # Rename columns to match OHLC format
    candle_df.rename(columns={'first': 'open', 'max': 'high', 'min': 'low', 'last': 'close'}, inplace=True)

    # Interpolate NaN values in OHLC columns
    candle_df[['open', 'high', 'low', 'close']] = candle_df[['open', 'high', 'low', 'close']].interpolate()

    # Calculate mode of 'fearGreedIndex' for each interval
    mode_fg = df.resample(interval, on='local_time')['fearGreedIndex'].agg(lambda x: x.values.tolist())

    # Sum volume within each interval
    volume_sum = df.resample(interval, on='local_time')['volume24h'].sum()

    # Extract the last local_time value in each interval
    last_time = df.resample(interval, on='local_time')['local_time'].last()

    # Merge mode, volume, and correct timestamp into candlestick data
    candle_df['fearGreedIndex'] = mode_fg
    candle_df['volume'] = volume_sum

    # Reset index for clean DataFrame structure


    candle_df['time'] = last_time
    candle_df.reset_index(inplace=True)
    candle_df.drop(columns=['local_time'], inplace=True)

    # Compute 24-hour forward price for each timestamp
    t = pd.Timedelta(hours=24*4)
    r = pd.Timedelta(minutes=5)

    comparing_price = "close"

    def compute_future_values(x):
        future_data = df[(df["local_time"] <= (x + t)) & (df["local_time"] > x + r)]

        if future_data.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, {}

        future_max = future_data["priceUSD"].max()
        future_min = future_data["priceUSD"].min()
        future_close = future_data["priceUSD"].iloc[-1]
        future_avg = future_data["priceUSD"].mean()

        # Find corresponding timestamps
        future_max_time = future_data.loc[future_data["priceUSD"].idxmax(), "local_time"]
        future_min_time = future_data.loc[future_data["priceUSD"].idxmin(), "local_time"]
        future_close_time = future_data["local_time"].iloc[-1]

        close = candle_df[candle_df["time"] == x].iloc[0][comparing_price]
        threshold_comparison = future_data["priceUSD"] / close

        # Find first occurrence of priceUSD going below each max_pain threshold
        below_threshold_times = {value: future_data[threshold_comparison < value]["local_time"].min()
        if not future_data[threshold_comparison < value].empty else np.nan
                                 for value in max_pains}

        future_prior_to_max = df[(df["local_time"] <= future_max_time) & (df["local_time"] > x + r)]
        future_min_prior_to_max = future_prior_to_max["priceUSD"].min()
        future_min_prior_to_max_time = future_data.loc[future_data["priceUSD"].idxmin(), "local_time"]

        return future_max, future_min, future_close, future_avg, future_max_time, future_min_time, future_close_time, below_threshold_times, future_min_prior_to_max, future_min_prior_to_max_time

    # Apply function once instead of multiple searches
    candle_df[['future_price', 'future_price_min', 'future_price_close', 'future_price_avg',
               'future_price_max_time', 'future_price_min_time', 'future_price_close_time', 'drop_below_times',
               'future_min_prior_to_max', 'future_min_prior_to_max_time']] = \
    candle_df['time'].apply(
        lambda x: pd.Series(compute_future_values(x))
    )

    # Compute log returns in vectorized form after extraction
    candle_df['log_returns'] = np.log(candle_df['future_price'] / candle_df[comparing_price])
    candle_df['log_returns_min'] = np.log(candle_df['future_price_min'] / candle_df[comparing_price])
    candle_df['log_returns_close'] = np.log(candle_df['future_price_close'] / candle_df[comparing_price])
    candle_df['log_returns_avg'] = np.log(candle_df['future_price_avg'] / candle_df[comparing_price])

    candle_df['log_returns_min_prior_to_max'] = np.log(candle_df['future_min_prior_to_max'] / candle_df[comparing_price])

    # Drop NaNs where future price was unavailable
    candle_df.dropna(subset=['log_returns', 'log_returns_min', 'log_returns_close', 'log_returns_avg'], inplace=True)

    candle_df["time"] = pd.to_datetime(candle_df["time"])

    return candle_df[["time", "open", "high", "low", "close", "volume", "fearGreedIndex",
                      "future_price", "future_price_min", "future_price_close", "future_price_avg",
                      "log_returns", "log_returns_min", "log_returns_close", "log_returns_avg",
                      'future_price_max_time', 'future_price_min_time', 'future_price_close_time', 'drop_below_times',
                      'log_returns_min_prior_to_max', 'future_min_prior_to_max_time'
                      ]]