from datetime import datetime, timedelta
from api_executor import ApiExecutor
from runner import Runner, Runnable
from datastream import DataStream
from livetrader import LiveTrader
from predictor import Predictor
from server import StatusServer  # Import the new StatusServer


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
    """

    def __init__(self, predictor, live_trader, data_stream: DataStream, available_coins):
        self.predictor = predictor
        self.live_trader = live_trader
        self.data_stream = data_stream
        self.available_coins = available_coins

    def tick(self):
        """
        This method is called by the Runner at the scheduled time.
        """
        print("\n--- Running Prediction Task ---")
        # In a live scenario, the predictor might need historical data.
        # We fetch the last 30 days for context.
        thirty_days_ago = datetime.now() - timedelta(days=300)
        df_per_coin = self.data_stream.get_historical_prices(earliest_date=thirty_days_ago)

        predicted_coins = self.predictor.make_predictions(
            available_coins=self.available_coins,
            df_per_coin=df_per_coin,
            timestamp=datetime.now()
        )

        self.data_stream.update_tracked_assets(predicted_coins)
        self.live_trader.process_predictions(predicted_coins)


def run_live_trading_simulation():
    """
    Sets up and runs the entire live trading simulation.
    """
    print("--- Starting Live Trading Simulation ---")

    # 1. Define the full universe of assets the predictor can choose from.
    asset_universe = [
        'airswap', 'chainlink', 'wax', 'decentraland', 'near',
        'nervos-network', 'floki-inu', 'arbitrum', 'open-campus',
        'bonk', 'sui', 'dogwifcoin'
    ]

    # 2. Initialize the main components of the trading system.
    # For live trading, set simulation_mode=False and provide API keys.
    api_executor = ApiExecutor(asset_universe_ids=asset_universe, simulation_mode=True)
    live_trader = LiveTrader(initial_budget=1000, api_executor=api_executor)
    data_stream = DataStream(asset_universe)
    predictor = Predictor(asset_universe)

    # 3. Set up the tasks that will be run by the Runner.
    prediction_task = PredictionTask(
        predictor=predictor,
        live_trader=live_trader,
        data_stream=data_stream,
        available_coins=asset_universe
    )
    data_stream_task = DataStreamTask(data_stream)

    # 4. Subscribe the LiveTrader to receive price updates from the DataStream.
    data_stream.subscribe(live_trader)

    # 5. Initialize and start the HTTP Status Server.
    # The server runs in a separate thread and does not block the main loop.
    status_server = StatusServer(live_trader=live_trader)
    status_server.start()

    # 6. Initialize the Runner, which orchestrates all scheduled tasks.
    runner = Runner()

    # 7. Add tasks to the Runner's schedule.
    runner.add_task(data_stream_task, interval_minutes=5, align_to_clock=True)
    runner.add_task(prediction_task, at_time="23:55")

    # 8. Perform an initial prediction to start the first cycle.
    print("\n--- Performing initial model prediction for demonstration ---")
    prediction_task.tick()

    # 9. Start the simulation loop.
    try:
        runner.loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        status_server.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    run_live_trading_simulation()
