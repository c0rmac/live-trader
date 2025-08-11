from live_trade.templ.datastream import PriceTick


class TradingAgent(object):

    def __init__(self):
        return

    def receive_next_price_tick(self, price_tick):
        pass


class Trade(TradingAgent):

    lifecycle_expired = False

    def __init__(self, expiration):
        super().__init__()
        self.expiration = expiration


class LiveTrader(TradingAgent):

    trade: Trade | None = None

    def receive_next_price_tick(self, price_tick: PriceTick):
        if self.trade is None:
            self.trade = PreTrade()

        price_action = self.trade.receive_next_price_tick(price_tick)

        if self.trade is PreTrade:
            if self.trade.lifecycle_expired:
                self.trade = None
            else:
                if price_action == "buy":
                    self.buy()

        elif self.trade is Trade:
            if self.trade.lifecycle_expired:
                self.trade = None
            else:
                if price_action == "sell":
                    self.sell()

        else:
            pass


    def buy(self):
        pass

    def sell(self):
        pass


class PreTrade(TradingAgent):

    def __init__(self):
        super().__init__()
        return

    def receive_next_price_tick(self, price_tick):
        return ["buy", "hold"]

class ActiveTrade(TradingAgent):

    def __init__(self, limit: float, stop: float):
        super().__init__()
        self.limit = limit
        self.stop = stop

    def receive_next_price_tick(self, price_tick):

        return ["sell", "hold"]