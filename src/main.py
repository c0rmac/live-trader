from datetime import datetime, timedelta

from api_executor import ApiExecutor
from runner import Runner, Runnable
from datastream import DataStream
from livetrader import LiveTrader
from predictor import Predictor  # Assuming these files exist as per the original main.py


class DataStreamTask(Runnable):
    """A Runnable wrapper for the DataStream to make it compatible with the Runner."""

    def __init__(self, data_stream: DataStream):
        self.data_stream = data_stream

    def tick(self):
        """This method is called by the Runner to fetch new data."""
        self.data_stream.fetch_and_notify()


class PredictionTask(Runnable):
    """
    A wrapper class that makes the Predictor compatible with the Runner.
    This class is responsible for triggering predictions on a schedule.
    """

    def __init__(self, predictor, live_trader, data_stream: DataStream, available_coins):
        self.predictor = predictor
        self.live_trader = live_trader
        self.data_stream = data_stream
        self.available_coins = available_coins

    def tick(self):
        """
        This method is called by the Runner at the scheduled time.
        It generates predictions and updates the DataStream and LiveTrader.
        """
        print("\n--- Running Prediction Task ---")
        # In a live scenario, the predictor might need historical data to make a decision.
        # We fetch it here. Let's fetch the last 30 days for context.
        thirty_days_ago = datetime.now() - timedelta(days=300)
        df_per_coin = self.data_stream.get_historical_prices(earliest_date=thirty_days_ago)

        predicted_coins = self.predictor.make_predictions(
            available_coins=self.available_coins,
            df_per_coin=df_per_coin,
            timestamp=datetime.now()
        )

        print(f"Prediction task resulted in the following assets: {predicted_coins}")

        # --- This is the key change ---
        # Update the DataStream to track the newly predicted assets.
        print("Updating DataStream with new assets to track...")
        self.data_stream.update_tracked_assets(predicted_coins)

        # Pass the new predictions to the trader to start a new cycle if one isn't active.
        self.live_trader.process_predictions(predicted_coins)


def run_live_trading_simulation():
    """
    Sets up and runs the entire live trading simulation.
    """
    print("--- Starting Live Trading Simulation ---")

    # 1. Define the full universe of assets the predictor can choose from.
    asset_universe = [
        'airswap', # Coinbase
        'chainlink', # Binance / MEXC
        'wax', # Binance / MEXC
        'decentraland', # Binance / MEXC
        'near', # Binance / MEXC
        'nervos-network', # Binance / MEXC
        'floki-inu', # Binance / MEXC
        'arbitrum', # Binance / MEXC
        'open-campus', # Binance / MEXC
        'bonk', # Binance / MEXC
        'sui', # Binance / MEXC
        'dogwifcoin' # Binance / MEXC
    ]

    # 2. Initialize the main components of the trading system.
    api_executor = ApiExecutor(asset_universe)
    live_trader = LiveTrader(initial_budget=1000, api_executor=api_executor)
    data_stream = DataStream(asset_universe)  # Initialized empty, will be populated by PredictionTask
    predictor = Predictor(asset_universe)

    # 3. Set up the tasks that will be run by the Runner.
    prediction_task = PredictionTask(
        predictor=predictor,
        live_trader=live_trader,
        data_stream=data_stream,
        available_coins=asset_universe
    )
    # This task wraps the datastream's fetch call to make it runnable.
    data_stream_task = DataStreamTask(data_stream)

    # 4. Subscribe the LiveTrader to receive price updates from the DataStream.
    data_stream.subscribe(live_trader)

    # 5. Initialize the Runner, which orchestrates all scheduled tasks.
    runner = Runner()

    # 6. Add tasks to the Runner's schedule.
    # The DataStream will "tick" every 5 minutes (300 seconds).
    runner.add_task(data_stream_task, interval_minutes=1, align_to_clock=True)

    # The PredictionModel will run once a day at 23:55.
    runner.add_task(prediction_task, at_time="23:55")

    # 7. Perform an initial prediction to set the initial assets to track.
    print("\n--- Performing initial model prediction for demonstration ---")
    prediction_task.tick()

    # 8. Start the simulation loop.
    # The runner will now continuously run the scheduled tasks.
    runner.loop()


if __name__ == "__main__":
    # To run the live simulation, execute this file.
    run_live_trading_simulation()
