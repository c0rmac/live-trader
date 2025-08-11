from live_trade.templ.runner import Runnable


class DataStream(Runnable):

    def fetch_coin_price(self):

        return

    def tick(self):

        return

    def run(self):

        return

    def subscribe(self):

        return


class PriceTick:

    def __init__(self, price, datetime):
        self.price = price
        self.datetime = datetime