import random
import time
# In a real application, you would install and import the ccxt library:
# pip install ccxt
import ccxt


class ApiExecutor:
    """
    A class to handle the execution of trades with external exchange APIs.
    This class is designed to be a realistic implementation using the ccxt library structure.
    """
    ALLOWED_EXCHANGES = ['binance', 'coinbase', 'kraken', 'kucoin']

    def __init__(self, asset_universe_ids=None, api_keys=None, simulation_mode=True, trading_fee_percent=0.001,
                 slippage_percent=0.0005):
        """
        Initializes the ApiExecutor.

        :param asset_universe_ids: A list of coin IDs (e.g., ['bitcoin', 'ethereum']).
        :param api_keys: A dictionary of API keys, e.g., {'binance': {'apiKey': '...', 'secret': '...'}}.
        :param simulation_mode: If True, simulates trades. If False, attempts to make real API calls.
        """
        self.simulation_mode = simulation_mode
        self.TRADING_FEE_PERCENT = trading_fee_percent
        self.SLIPPAGE_PERCENT = slippage_percent
        self.exchanges = {}
        self.asset_universe = {}  # This will be built dynamically
        self.coin_id_to_symbol = {}  # Cache for converting IDs to symbols

        self._connect_to_exchanges(api_keys or {})
        self._build_asset_universe(asset_universe_ids or [])

    def _connect_to_exchanges(self, api_keys):
        """
        Creates and authenticates connections to the allowed exchanges.
        """
        print(f"Connecting to exchanges in {'Simulation' if self.simulation_mode else 'Live'} mode...")
        for exchange_name in self.ALLOWED_EXCHANGES:
            if not self.simulation_mode:
                if exchange_name not in api_keys:
                    print(f"  - WARNING: No API keys provided for {exchange_name}. Cannot connect.")
                    continue
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class(api_keys[exchange_name])
                    self.exchanges[exchange_name].load_markets()
                    print(f"  - Successfully connected to {exchange_name} (Live)")
                except Exception as e:
                    print(f"  - FAILED to connect to {exchange_name}: {e}")
            else:
                # For simulation, we create placeholder objects with sample market data
                exchange_obj = type('Exchange', (object,), {})()
                # --- Updated Simulation Markets ---
                # This now reflects the asset list provided by the user.
                simulated_markets = {
                    'binance': ['LINK/USDT', 'WAXP/USDT', 'MANA/USDT', 'NEAR/USDT', 'CKB/USDT', 'FLOKI/USDT',
                                'ARB/USDT', 'EDU/USDT', 'BONK/USDT', 'SUI/USDT', 'WIF/USDT'],
                    'coinbase': ['AST/USDT', 'LINK/USDT'],
                    'kraken': [],  # No specific assets from the list are primarily here in the simulation
                    'kucoin': []  # No specific assets from the list are primarily here in the simulation
                }
                exchange_obj.markets = {pair: {} for pair in simulated_markets.get(exchange_name, [])}
                self.exchanges[exchange_name] = exchange_obj
        print("Connections established.")

    def _get_symbol_from_id(self, asset_id):
        """
        Converts a coin ID (e.g., 'bitcoin') to its trading symbol (e.g., 'BTC').
        In a real application, this could use a service like CoinGecko or a local database.
        """
        if asset_id in self.coin_id_to_symbol:
            return self.coin_id_to_symbol[asset_id]

        # Expanded mapping to include all assets from the user's list
        id_to_symbol_map = {
            'airswap': 'AST', 'chainlink': 'LINK', 'wax': 'WAXP', 'decentraland': 'MANA',
            'near': 'NEAR', 'nervos-network': 'CKB', 'floki-inu': 'FLOKI', 'arbitrum': 'ARB',
            'open-campus': 'EDU', 'bonk': 'BONK', 'sui': 'SUI', 'dogwifcoin': 'WIF'
        }
        symbol = id_to_symbol_map.get(asset_id.lower())
        if symbol:
            self.coin_id_to_symbol[asset_id] = symbol
        return symbol

    def _build_asset_universe(self, asset_universe_ids):
        """
        Builds the internal asset universe by checking which of our connected
        exchanges trade the provided list of assets.
        """
        print("Building asset universe...")
        for asset_id in asset_universe_ids:
            symbol = self._get_symbol_from_id(asset_id)
            if not symbol:
                continue

            trading_pair = f"{symbol.upper()}/USDT"
            self.asset_universe[asset_id] = []

            for exchange_name, exchange_obj in self.exchanges.items():
                if hasattr(exchange_obj, 'markets') and trading_pair in exchange_obj.markets:
                    self.asset_universe[asset_id].append(exchange_name)
        print("Asset universe built.")

    def _find_exchanges_for_coin(self, asset_id):
        """
        Finds out which exchanges list a particular coin by looking up the asset universe.
        """
        return self.asset_universe.get(asset_id, [])

    def execute_trade(self, asset_id, action, intended_amount, market_price):
        """
        The main method for executing a trade. It finds a suitable exchange
        and then executes the trade, returning the results.
        """
        available_exchanges = self._find_exchanges_for_coin(asset_id)

        chosen_exchange = None
        for preferred_exchange in self.ALLOWED_EXCHANGES:
            if preferred_exchange in available_exchanges and self.exchanges.get(preferred_exchange):
                chosen_exchange = preferred_exchange
                break

        if not chosen_exchange:
            return {"success": False, "error": f"No allowed exchange found for {asset_id}"}

        return self._execute_order_on_exchange(chosen_exchange, asset_id, action, intended_amount, market_price)

    def _execute_order_on_exchange(self, exchange_name, asset_id, action, intended_amount, market_price):
        """
        Places an order on a specific exchange and returns the result.
        """
        if self.simulation_mode:
            return self._simulate_trade(action, intended_amount, market_price)

        # --- Real World Logic (using ccxt structure) ---
        exchange = self.exchanges[exchange_name]
        symbol = self.coin_id_to_symbol.get(asset_id)
        if not symbol:
            return {"success": False, "error": f"Could not find symbol for {asset_id}"}

        trading_pair = f"{symbol.upper()}/USDT"
        print(f"  -> LIVE API Call: Placing {action} order for {trading_pair} on {exchange_name}.")

        try:
            if action == 'buy':
                order = exchange.create_market_buy_order_with_cost(trading_pair, intended_amount)
            elif action == 'sell':
                order = exchange.create_market_sell_order(trading_pair, intended_amount)
            else:
                return {"success": False, "error": "Invalid action"}

            return self._parse_order_result(order, action)

        except ccxt.InsufficientFunds as e:
            print(f"  -> LIVE API ERROR on {exchange_name}: {e}")
            return {"success": False, "error": "Insufficient funds"}
        except Exception as e:
            print(f"  -> LIVE API ERROR on {exchange_name}: {e}")
            return {"success": False, "error": str(e)}

    def _parse_order_result(self, order, action):
        """
        Parses the complex order dictionary returned by ccxt into a simple, consistent format.
        """
        if not order or 'id' not in order:
            return {"success": False, "error": "Invalid order object returned from exchange"}

        net_proceeds = None
        if action == 'sell':
            net_proceeds = order.get('cost', 0) - order.get('fee', {}).get('cost', 0)

        return {
            "success": True,
            "amount_executed": order.get('filled'),
            "execution_price": order.get('average'),
            "fee": order.get('fee', {}).get('cost'),
            "net_proceeds": net_proceeds
        }

    def _simulate_trade(self, action, intended_amount, market_price):
        """
        Simulates a trade execution with fees and slippage.
        """
        if action == 'buy':
            execution_price = market_price * (1 + self.SLIPPAGE_PERCENT)
            capital_to_spend = intended_amount
            fee = capital_to_spend * self.TRADING_FEE_PERCENT
            capital_after_fee = capital_to_spend - fee
            amount_executed = capital_after_fee / execution_price
            return {"success": True, "amount_executed": amount_executed, "execution_price": execution_price, "fee": fee}

        if action == 'sell':
            execution_price = market_price * (1 - self.SLIPPAGE_PERCENT)
            amount_to_sell = intended_amount
            gross_proceeds = amount_to_sell * execution_price
            fee = gross_proceeds * self.TRADING_FEE_PERCENT
            net_proceeds = gross_proceeds - fee
            return {"success": True, "net_proceeds": net_proceeds, "execution_price": execution_price, "fee": fee}

        return {"success": False}
