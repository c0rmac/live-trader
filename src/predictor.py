import pickle
import random
import statistics

import numpy as np
import pandas as pd

from breakout_detection3 import dataloader, datapreprocess
from live_trade.tools import generate_candles


class Predictor:
    """
    A class to encapsulate prediction logic for the trading strategy.
    """

    def load_transformed_coins_pickle(self, file_path):
        """Loads transformed coins from a pickle file."""
        try:
            with open(file_path, "rb") as f:
                transformed_coins = pickle.load(f)
            print(f"Loaded transformed coins from {file_path}")
            return transformed_coins
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None

    def extract_data(self, name):
        optimal_cluster = self.load_transformed_coins_pickle(f"../models/{name}_cluster.pkl")
        models = self.load_transformed_coins_pickle(f"../models/{name}_model.pkl")

        return optimal_cluster, models

    def assign_clusters(self, test_candle_df, optimal_cluster):
        kmeans = optimal_cluster['kmeans']
        selected_cols = optimal_cluster['selected_cols']
        scaler = optimal_cluster['scaler']

        X_test = test_candle_df[selected_cols].dropna()

        X_test_scaled = scaler.transform(X_test)  # Use same scaler!
        test_labels = kmeans.predict(X_test_scaled)

        # Store labels in test_df
        # test_labels = pd.Series(test_labels, index=X_test.index, name="cluster")
        # test_candle_df = pd.concat([train_candle_df.drop(columns=["cluster"], errors='ignore'), test_labels], axis=1)
        test_candle_df.loc[X_test.index, "cluster"] = test_labels

    def assign_trade_signal(self, test_candle_df, models, optimal_cluster):
        grouped_test_candle_dfs = {cluster: group for cluster, group in test_candle_df.groupby('cluster')}

        for cluster, candle_test in grouped_test_candle_dfs.items():
            if cluster in models:
                model = models[cluster]
                train_breakthrough_ratios = optimal_cluster["train_breakthrough_ratios"][int(cluster)]
                train_loss_ratios = optimal_cluster["train_loss_ratios"][int(cluster)]

                candle_test["trade"] = model.predict_from_val_threshold(candle_test, 1, 0.85)
                # candle_test["trade"] = model.predict(candle_test)
                candle_test["returns"] = np.exp(candle_test["log_returns"])
                candle_test["max_loss"] = np.exp(candle_test["log_returns_min"])
                candle_test["max_loss_prior_to_max"] = np.exp(candle_test["log_returns_min_prior_to_max"])
                candle_test["close_returns"] = np.exp(candle_test["log_returns_close"])
                candle_test["avg_returns"] = np.exp(candle_test["log_returns_avg"])

        return grouped_test_candle_dfs

    def identify_trade_clusters(self, grouped_test_candle_dfs, cluster_day_gap):
        trades = []
        cluster_df = pd.concat(grouped_test_candle_dfs.values(), ignore_index=True)
        cluster_df = cluster_df.sort_values(by='time', ascending=True)

        if "trade" not in cluster_df.columns:
            return trades

        cluster_df = cluster_df[cluster_df["trade"] == True]
        # Sort by time, just in case
        cluster_df = cluster_df.sort_values("time").reset_index(drop=True)

        # Calculate time difference in days
        time_diff = cluster_df["time"].diff().dt.days.fillna(0)

        # Start a new group when the time jump is 1 day or more
        cluster_df["group_id"] = (time_diff >= cluster_day_gap + 1).cumsum()

        # Make sure 'time' is a datetime type
        cluster_df["time"] = pd.to_datetime(cluster_df["time"])

        # Group and plot
        grouped = cluster_df.groupby("group_id")

        for group_id, group_data in grouped:
            trades.append(group_data)

        return trades

    def find_all_trades(self, test_candle_df, cluster_day_gap, models, optimal_cluster):
        # name = "ong"
        # file_name = "../coins/ong_565.pkl"
        # name = "iotex"
        # file_name = "../coins/iotex_565.pkl"

        # file_name = "../coins/88mph_511.pkl"
        # name = "88mph"

        self.assign_clusters(test_candle_df, optimal_cluster)
        grouped_test_candle_dfs = self.assign_trade_signal(test_candle_df, models, optimal_cluster)

        trades = self.identify_trade_clusters(grouped_test_candle_dfs, cluster_day_gap)
        return trades

    def __init__(self, coins):
        """
        Initializes the Predictor. In a real-world scenario, this is where
        you might load a pre-trained machine learning model.
        """

        optimal_cluster_per_coin = {}
        models_per_coin = {}

        for name in coins:

            optimal_cluster, models = self.extract_data(name)

            optimal_cluster_per_coin[name] = optimal_cluster
            models_per_coin[name] = models

        self.optimal_cluster_per_coin = optimal_cluster_per_coin
        self.models_per_coin = models_per_coin

    def make_predictions(self, available_coins, df_per_coin, timestamp, candle_df_per_coin=None):
        """
        Generates a list of coins to trade based on some logic.

        In a real-world scenario, this would involve complex analysis of the
        historical data contained in df_per_coin up to the given timestamp.

        For this simulation, we'll just randomly select a subset of the
        available coins to trade.

        :param available_coins: A list of all coin names available for trading.
        :param df_per_coin: A dictionary of DataFrames with historical data for each coin.
        :param timestamp: The current timestamp of the backtest.
        :return: A list of coin names predicted for trading.
        """
        if not available_coins:
            return []

        trade_signals = {}

        for slug, df in df_per_coin.items():
            if candle_df_per_coin is None:
                candle_df = generate_candles(df)
                candle_df = candle_df[(candle_df['time'].dt.hour == 23) & (candle_df['time'].dt.minute >= 45) & (candle_df['time'].dt.minute <= 55)]

                candle_df = dataloader.datapre(candle_df)
                candle_df["fearGreedIndex"] = candle_df["fearGreedIndex"].apply(
                    lambda x: statistics.mode(x) if isinstance(x, list) else x)

                candle_df = candle_df.tail(70)
                candle_df, fear_greed_trail = datapreprocess.generate_lookback(candle_df)
                candle_df = candle_df.dropna()
                candle_df = candle_df.tail(1)

                optimal_cluster = self.optimal_cluster_per_coin[slug]
                models = self.models_per_coin[slug]

                kmeans = optimal_cluster['kmeans']
                selected_cols = optimal_cluster['selected_cols']
                scaler = optimal_cluster['scaler']

                X_test = candle_df[selected_cols].dropna()

                X_test_scaled = scaler.transform(X_test)  # Use same scaler!
                cluster = kmeans.predict(X_test_scaled)
                cluster = cluster[0]

                if cluster in models:
                    model = models[cluster]

                    trade_signal = model.predict_from_val_threshold(candle_df, 1, 0.85)[0]
                    trade_signals[slug] = trade_signal
                else:
                    trade_signals[slug] = False
            else:
                candle_df = candle_df_per_coin[slug]

                candle_df = candle_df.tail(1)
                fear_greed_index = candle_df["mode_fearGreedIndex_step0"].values[0]

                trade_signals[slug] = candle_df.iloc[0]["trade"]


        nominated_coins = [slug for slug, trade in trade_signals.items() if trade_signals[slug] == True]

        if not nominated_coins:
            return []

        # Simulate a model by randomly selecting a number of coins to trade.
        if len(nominated_coins) > 2:
            num_to_select = random.randint(
                2, min(4, len(nominated_coins))
            )
        else:
            num_to_select = len(nominated_coins)

        num_to_select = min(5, len(nominated_coins))
        predicted_coins = random.sample(nominated_coins, k=num_to_select)

        #num_to_select = min(5, len(available_coins))
        #predicted_coins = random.sample(available_coins, k=num_to_select)

        return predicted_coins

