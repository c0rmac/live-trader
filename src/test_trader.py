import statistics
import warnings

import pandas as pd
import pickle
import os
import random
from datetime import datetime, timedelta, time

from data import dataloader, datapreprocess
from api_executor import ApiExecutor
from tools import generate_candles, generate_candles_with_indicators
from livetrader import LiveTrader
from datastream import PriceTick
from predictor import Predictor  # Import the new Predictor class
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

# Wrapper for tqdm + joblib
class TqdmParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc="Simulations in progress", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_tqdm = use_tqdm
        self.total = total
        self.desc = desc

    def __call__(self, *args, **kwargs):
        if self.use_tqdm:
            with tqdm(total=self.total, desc=self.desc) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)
        else:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if hasattr(self, '_pbar'):
            self._pbar.update()

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
        coin_name = coin_name.split('_')[0]
        with open(filepath, 'rb') as f:
            df = pickle.load(f)
            # Ensure 'local_time' is datetime
            df['local_time'] = pd.to_datetime(df['local_time'])
            df_per_coin[coin_name] = df

    print(f"Loaded data for {list(df_per_coin.keys())} from specified files.")
    return df_per_coin

def run_simulation(price_data_for_iteration, sentiment_data_for_iteration, candle_df_per_coin, model_run_time, predictor, available_coins_for_prediction, df_per_coin, api_executor):
    last_prediction_day = None

    trader = LiveTrader(initial_budget=1000, api_executor=api_executor)

    for timestamp, prices in price_data_for_iteration.iterrows():
        # Update latest prices for all coins at this timestamp before making decisions
        for asset_id, price in prices.items():
            if pd.notna(price):
                trader.latest_prices[asset_id] = price

        # Process the price tick for each coin at this timestamp
        for i, (asset_id, price) in enumerate(prices.items()):
            sentiment_data = sentiment_data_for_iteration[sentiment_data_for_iteration.index == timestamp]
            fear_greed_index = sentiment_data[asset_id].values[0]
            if pd.notna(price):
                price_tick = PriceTick(asset_id, price, timestamp, fear_greed_index)
                trader.receive_next_price_tick(price_tick, current_time=timestamp)

        trader.last_price_tick_submitted(current_time=timestamp)

        # Check if it's time to run the daily model prediction.
        current_day = timestamp.date()
        if timestamp.time() >= model_run_time and current_day != last_prediction_day:
            # Use the Predictor to make predictions
            #df_per_coin_cutoff = {slug : df[df["local_time"] <= timestamp] for slug, df in df_per_coin.items()}
            candle_df_per_coin_cutoff = {slug : df[df["time"] <= timestamp] for slug, df in candle_df_per_coin.items()}

            predicted_coins = predictor.make_predictions(
                available_coins=available_coins_for_prediction,
                df_per_coin=df_per_coin,
                timestamp=timestamp,
                candle_df_per_coin=candle_df_per_coin_cutoff
            )
            trader.process_predictions(predicted_coins, current_time=timestamp)
            last_prediction_day = current_day

    # 5. Print final results
    #print("\n--- Backtest Finished ---")
    #final_budget = trader.budget + trader.divestment_pool
    #print(f"Final Total Value: ${final_budget:,.2f}")
    #print(f"Net Profit/Loss: ${final_budget - trader.initial_budget:,.2f}")

    # 6. Get and display the structured log and summary tables
    log_df = trader.logger.get_log_df()
    profit_table = trader.logger.get_profit_table()
    cycle_returns_table = trader.logger.get_cycle_returns_table()

    _cycle_returns_table = cycle_returns_table[cycle_returns_table["Was Dry Run"] == False]

    #print("\n--- Cycle Returns ---")
    #print(cycle_returns_table)

    #print("\n--- Profit Table per Asset ---")
    #print(profit_table)

    # print("\n--- Full Trade Log ---")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    #     print(log_df)

    return _cycle_returns_table, cycle_returns_table

def run_backtest():
    """
    Initializes and runs the trading simulation against historical data,
    with daily predictions.
    """
    print("--- Starting Backtest ---")

    # 1. Define data directory and which coins to load for the test
    DATA_DIRECTORY = '../coins2'
    COINS_TO_TEST = [
        ('airswap', '669'),
        ('chainlink', '669'),
        ('wax', '669'),
        ('decentraland', '669'),
        ('near', '626'),
        ('nervos-network', '669'),
        ('floki-inu', '529'),
        ('arbitrum', '302'),
        ('open-campus', '289'),
        ('bonk', '332'),
        ('pepe', '293'),
        ('sui', '287'),
        ('dogwifcoin', '207')
    ]

    # Construct the full file paths for the chosen coins
    filepaths_to_load = [os.path.join(DATA_DIRECTORY, f"{coin[0]}_{coin[1]}.pkl") for coin in COINS_TO_TEST]

    # Load data only from the specified files, assuming they exist
    df_per_coin = load_data_from_pkls(file_paths=filepaths_to_load)

    if not df_per_coin:
        print("No data loaded. Aborting backtest.")
        return

    # 2. Initialize the LiveTrader and the Predictor
    predictor = Predictor([coin[0] for coin in COINS_TO_TEST])

    # 3. Prepare a unified DataFrame for easy iteration.
    # The index will be time, and columns will be the coin prices.
    def round_to_nearest_5min(ts):
        # Convert to pandas datetime if not already
        ts = pd.to_datetime(ts)
        # Round up/down to nearest 5 minutes
        return (ts.dt.floor('5min'))

    # Apply to each DataFrame in your loop
    prepared_dfs = []
    prepared_dfs2 = []

    for coin, df in df_per_coin.items():
        temp_df = df[['local_time', 'priceUSD']].copy()
        temp_df2 = df[['local_time', 'fearGreedIndex']].copy()

        # Round time to nearest 5 minutes
        temp_df['local_time'] = pd.to_datetime(temp_df['local_time']).dt.ceil('5min')
        temp_df2['local_time'] = pd.to_datetime(temp_df['local_time']).dt.ceil('5min')

        # Set index
        temp_df.set_index('local_time', inplace=True)
        temp_df2.set_index('local_time', inplace=True)

        # If duplicates exist after rounding, aggregate by mean (or keep first)
        temp_df = temp_df.groupby(temp_df.index).last()
        temp_df2 = temp_df2.groupby(temp_df2.index).last()

        temp_df.rename(columns={'priceUSD': coin}, inplace=True)
        temp_df2.rename(columns={'fearGreedIndex': coin}, inplace=True)

        prepared_dfs.append(temp_df)
        prepared_dfs2.append(temp_df2)

    # Merge and align all coins
    price_data_for_iteration = pd.concat(prepared_dfs, axis=1).sort_index()
    price_data_for_iteration.ffill(inplace=True)
    price_data_for_iteration.dropna(inplace=True)

    price_data_for_iteration = price_data_for_iteration[price_data_for_iteration.index >= "2025-01-01"]

    sentiment_data_for_iteration = pd.concat(prepared_dfs2, axis=1).sort_index()
    sentiment_data_for_iteration.ffill(inplace=True)
    sentiment_data_for_iteration.dropna(inplace=True)

    sentiment_data_for_iteration = sentiment_data_for_iteration[sentiment_data_for_iteration.index >= "2025-01-01"]

    # 4. Loop through the unified historical data, triggering daily predictions
    print("\n--- Processing Price Data ---")
    model_run_time = time(23, 55)

    available_coins_for_prediction = list(df_per_coin.keys())

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    candle_df_per_coin = {key : generate_candles_with_indicators(df) for key, df in df_per_coin.items()}

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


    api_executor = ApiExecutor([value[0] for value in COINS_TO_TEST])

    # Joblib delayed call
    def run_simulation_joblib(i):
        return run_simulation(
            price_data_for_iteration=price_data_for_iteration,
            sentiment_data_for_iteration=sentiment_data_for_iteration,
            candle_df_per_coin=candle_df_per_coin,
            model_run_time=model_run_time,
            predictor=predictor,
            available_coins_for_prediction=available_coins_for_prediction,
            df_per_coin=candle_df_per_coin,
            api_executor=api_executor
        )

    # run_simulation_joblib(0)

    # Execution
    returns = TqdmParallel(n_jobs=-1, total=150, desc="Running simulations")(
        delayed(run_simulation_joblib)(i) for i in range(150)
    )

    return_values_1 = np.array([_cycle_returns_table['End Capital'].max() for (_cycle_returns_table, _) in returns])
    return_values_2 = np.array([_cycle_returns_table['End Capital'].iloc[-1] for (_cycle_returns_table, _) in returns])

    mean = return_values_2.mean()
    std = return_values_2.std()

    # Plot the histogram
    plt.hist(return_values_1, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot the histogram
    plt.hist(return_values_2, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    returns

if __name__ == "__main__":
    # Note: This script assumes the required .pkl files
    # exist in the specified DATA_DIRECTORY.
    run_backtest()
