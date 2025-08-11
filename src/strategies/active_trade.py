from strategies.base_strategy import TradingAgent


class ActiveTrade(TradingAgent):
    """
    Manages an asset that has been purchased and is being actively traded.
    Includes logic for profit-taking, stop-loss, and dynamic stop adjustment.
    """

    def __init__(self, purchase_price, asset_id, limit_returns = 1.52,
                 #stop_returns = 0.95
                 stop_returns = 0.983
                 ):
        self.purchase_price = purchase_price
        self.asset_id = asset_id
        self.limit = purchase_price * limit_returns
        self.stop = purchase_price * stop_returns
        self.threshold_2_met = False
        self.tick_max = 0.0  # Track the peak return of the trade

    def receive_next_price_tick(self, price_tick, current_time=None):
        """
        Determines if the asset should be sold or held.
        Returns a tuple of (action, trigger_price).
        """
        current_price = price_tick.price
        initial_tick_return = current_price / self.purchase_price

        # If the trade hits a new high-water mark, reset the threshold.
        if initial_tick_return > self.tick_max:
            self.tick_max = initial_tick_return
            self.threshold_2_met = False

        # Dynamically adjust the stop-loss upwards if the trade is profitable
        if 1.5 > initial_tick_return > 1.1 and not self.threshold_2_met:
            _diff = 0.048
            #_diff = 0.07
            new_stop_return = max(initial_tick_return - _diff, 1.0)
            self.stop = self.purchase_price * new_stop_return
            self.threshold_2_met = True

        # Return the action and the price that triggered it
        if current_price > self.limit:
            return "sell_profit", self.limit
        if current_price < self.stop:
            return "sell_divest", self.stop

        return "hold", None

