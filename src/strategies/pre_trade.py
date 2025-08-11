from datetime import datetime, timedelta
from strategies.base_strategy import TradingAgent

class PreTrade(TradingAgent):
    """
    Manages an asset that is nominated for trading but not yet purchased.
    Triggers a 'buy' signal if the price moves significantly up or down.
    """
    def __init__(self, initial_price, expiration_hours=24, start_time=None, limit_returns = 1.005):
        self.initial_price = initial_price
        self.creation_time = start_time or datetime.now()
        self.expiration_time = self.creation_time + timedelta(hours=expiration_hours)
        self.limit_price = initial_price * limit_returns
        #self.limit_price = initial_price * 1.01
        self.stop_price = initial_price * 0.92
        #self.stop_price = initial_price * 0.91
        # print(f"    - PreTrade created at {self.creation_time}. Expires at {self.expiration_time}")

    def is_expired(self, current_time=None):
        """Checks if the pre-trade window has expired."""
        now = current_time or datetime.now()
        return now > self.expiration_time

    def receive_next_price_tick(self, price_tick, current_time=None):
        """Determines if the asset should be bought or held."""
        current_price = price_tick.price
        if current_price > self.limit_price or current_price < self.stop_price:
            return "buy", None
        return "hold", None

