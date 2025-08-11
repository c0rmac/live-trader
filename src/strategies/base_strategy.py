from datetime import datetime

class TradingAgent:
    """Base class for all trading strategy components."""
    def receive_next_price_tick(self, price_tick, current_time=None):
        """
        Processes the next price tick and returns an action.

        :param price_tick: A PriceTick object with asset data.
        :param current_time: The current time for backtesting purposes.
        :return: A string representing the desired action (e.g., "buy", "sell", "hold").
        """
        raise NotImplementedError

