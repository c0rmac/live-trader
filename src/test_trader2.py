import statistics
import warnings

import pandas as pd
import pickle
import os
import random
from datetime import datetime, timedelta, time

# Assuming these are custom user modules that exist in the environment
# from breakout_detection3 import dataloader, datapreprocess
from tools import generate_candles, generate_candles_with_indicators

from livetrader import LiveTrader
from datastream import PriceTick
from predictor import Predictor


def load_data_from_pkls(file_paths):
    """
    Loads a dictionary of DataFrames from a specific list of pickle files.
    """
    df_per_coin = {}

    for filepath in file_paths:
        if not os.path.exists(filepath):
            print(f"Warning: Data file not found at '{filepath}'. Skipping.")
            continue

        coin_name = os.path.splitext(os.path.basename(filepath))[0]
        # Handle cases like 'airswap_669' -> 'airswap'
        if '_' in coin_name:
            coin_name = coin_name.split('_')[0]

        with open(filepath, 'rb') as f:
            df = pickle.load(f)
            df['local_time'] = pd.to_datetime(df['local_time'])
            df_per_coin[coin_name] = df

    print(f"Loaded data for {list(df_per_coin.keys())} from specified files.")
    return df_per_coin


def run_backtest():
    """
    Initializes and runs the trading simulation against historical data,
    with daily predictions.
    """
    print("--- Starting Backtest ---")

    # 1. Define data directory and which coins to load for the test
    DATA_DIRECTORY = '../coins2'
    COINS_TO_TEST = [
        ('airswap', '669'), ('chainlink', '669'), ('wax', '669'),
        ('decentraland', '669'), ('near', '626'), ('nervos-network', '669'),
        ('floki-inu', '529'), ('arbitrum', '302'), ('open-campus', '289'),
        ('bonk', '332'), ('pepe', '293'), ('sui', '287'), ('dogwifcoin', '207')
    ]

    filepaths_to_load = [os.path.join(DATA_DIRECTORY, f"{coin[0]}_{coin[1]}.pkl") for coin in COINS_TO_TEST]
    df_per_coin = load_data_from_pkls(file_paths=filepaths_to_load)

    if not df_per_coin:
        print("No data loaded. Aborting backtest.")
        return

    # 2. Initialize the LiveTrader and the Predictor
    trader = LiveTrader(initial_budget=1000)
    predictor = Predictor([coin[0] for coin in COINS_TO_TEST])

    # 3. Prepare DataFrames for efficient lookup
    for coin, df in df_per_coin.items():
        # Set index for fast .loc lookups
        df.set_index('local_time', inplace=True, drop=False)
        df.sort_index(inplace=True)

    # Choose a reference DataFrame to drive the simulation clock
    reference_coin_name = COINS_TO_TEST[0][0]
    reference_df = df_per_coin[reference_coin_name]
    reference_df = reference_df[reference_df.index >= "2025-01-01"]

    print(f"\nUsing '{reference_coin_name}' as the reference timeline for the backtest.")

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    candle_df_per_coin = {key: generate_candles_with_indicators(df) for key, df in df_per_coin.items()}

    for slug, candle_df in candle_df_per_coin.items():
        candle_df = candle_df[candle_df["time"] >= "2025-01-01"]
        optimal_cluster = predictor.optimal_cluster_per_coin[slug]
        models = predictor.models_per_coin[slug]

        kmeans = optimal_cluster['kmeans']
        selected_cols = optimal_cluster['selected_cols']
        scaler = optimal_cluster['scaler']

        X_test = candle_df[selected_cols].dropna()

        X_test_scaled = scaler.transform(X_test)  # Use same scaler!
        cluster = kmeans.predict(X_test_scaled)
        candle_df["cluster"] = cluster

        grouped_candle_dfs = {cluster: group for cluster, group in candle_df.groupby('cluster')}

        for cluster, candle_test in grouped_candle_dfs.items():
            if cluster in models:
                model = models[cluster]

                trade_signal = model.predict_from_val_threshold(candle_test, 1, 0.85)
                candle_df.loc[candle_test.index, "trade"] = trade_signal

        candle_df_per_coin[slug] = candle_df

    # 4. Loop through the reference timeline, looking up other prices as needed
    print("\n--- Processing Price Data ---")
    last_prediction_day = None
    model_run_time = time(23, 55)
    available_coins_for_prediction = list(df_per_coin.keys())

    # This loop iterates through the master clock provided by the reference coin
    for ref_timestamp, ref_row in reference_df.iterrows():

        # --- Update latest prices for all coins at this timestamp ---
        current_prices = {}
        for slug, df in df_per_coin.items():
            try:
                # Fast lookup if timestamp exists
                tick = df.loc[ref_timestamp]
            except KeyError:
                # Slower lookup for the most recent previous tick
                # This ensures we always have a price, even if data is not perfectly aligned
                temp_df = df[df['local_time'] <= ref_timestamp]
                if not temp_df.empty:
                    tick = temp_df.iloc[-1]
                else:
                    tick = None  # No data available for this coin yet

            if tick is not None:
                price = tick["priceUSD"]
                if pd.notna(price):
                    trader.latest_prices[slug] = price
                    current_prices[slug] = price

        # --- Check if it's time to run the daily model prediction ---
        current_day = ref_timestamp.date()
        if ref_timestamp.time() >= model_run_time and current_day != last_prediction_day:
            candle_df_per_coin_cutoff = {slug: df[df["time"] <= ref_timestamp] for slug, df in candle_df_per_coin.items()}

            predicted_coins = predictor.make_predictions(
                available_coins=available_coins_for_prediction,
                df_per_coin=df_per_coin,  # Pass full history for analysis
                timestamp=ref_timestamp,
                candle_df_per_coin=candle_df_per_coin_cutoff
            )
            trader.process_predictions(predicted_coins, current_time=ref_timestamp)
            last_prediction_day = current_day

        # --- Process the price tick for each coin at this timestamp ---
        for asset_id, price in current_prices.items():
            price_tick = PriceTick(asset_id, price, ref_timestamp)
            trader.receive_next_price_tick(price_tick, current_time=ref_timestamp)

    # 5. Print final results
    print("\n--- Backtest Finished ---")
    log_df = trader.logger.get_log_df()
    profit_table = trader.logger.get_profit_table()
    cycle_returns_table = trader.logger.get_cycle_returns_table()

    _cycle_returns_table = cycle_returns_table[cycle_returns_table["Was Dry Run"] == False]

    print("\n--- Cycle Returns ---")
    print(cycle_returns_table)

    print("\n--- Profit Table per Asset ---")
    print(profit_table)


if __name__ == "__main__":
    run_backtest()

