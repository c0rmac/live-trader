import statistics

import numpy as np
import pandas as pd

from data import dataloader, datapreprocess


def generate_candles(df: pd.DataFrame, interval: str = '1d'):
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

    candle_df["time"] = pd.to_datetime(candle_df["time"])

    return candle_df[["time", "open", "high", "low", "close", "volume", "fearGreedIndex"]]

def generate_candles_with_indicators(df):
    candle_df = generate_candles(df)
    candle_df = dataloader.datapre(candle_df)
    candle_df["fearGreedIndex"] = candle_df["fearGreedIndex"].apply(
        lambda x: statistics.mode(x) if isinstance(x, list) else x)

    candle_df, fear_greed_trail = datapreprocess.generate_lookback(candle_df)
    candle_df = candle_df.dropna()

    return candle_df