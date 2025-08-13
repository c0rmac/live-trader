from datetime import datetime, timedelta
import random
import math
import pandas as pd
from strategies.pre_trade import PreTrade
from strategies.active_trade import ActiveTrade
from logger import TradeLogger
from api_executor import ApiExecutor # Import the ApiExecutor

class Budget:
    """
    A class to manage and track the different capital pools for trading.
    It maintains a history of all allocations within a cycle.
    """

    def __init__(self, initial_capital=0.0):
        self.main = initial_capital
        # These dictionaries now hold the history of allocations for the entire cycle
        self.pre_trade_allocations = {}  # asset_id -> {'amount': float, 'status': str}
        self.active_trade_allocations = {}  # asset_id -> {'amount': float, 'status': str}
        self.divestment = 0.0
        self.held_for_reinvestment = 0.0

    def get_total(self):
        """
        Calculates the total capital by summing unallocated pools and all *active* allocations.
        """
        active_pre_trade_capital = sum(
            info['amount'] for info in self.pre_trade_allocations.values() if info['status'] == 'pending')
        active_trade_capital = sum(
            info['amount'] for info in self.active_trade_allocations.values() if info['status'] == 'active')

        return (self.main +
                active_pre_trade_capital +
                active_trade_capital +
                self.divestment +
                self.held_for_reinvestment)

    def allocate_to_pre_trade(self, amount, asset_id):
        if self.main >= amount or math.isclose(self.main, amount):
            self.main -= amount
            self.pre_trade_allocations[asset_id] = {'amount': amount, 'status': 'pending'}
            return True
        return False

    def pre_trade_to_active(self, asset_id):
        if asset_id in self.pre_trade_allocations and self.pre_trade_allocations[asset_id]['status'] == 'pending':
            amount = self.pre_trade_allocations[asset_id]['amount']
            self.pre_trade_allocations[asset_id]['status'] = 'converted_to_active'
            self.active_trade_allocations[asset_id] = {'amount': amount, 'status': 'active'}
            return True
        return False

    def pre_trade_to_divestment(self, asset_id):
        if asset_id in self.pre_trade_allocations and self.pre_trade_allocations[asset_id]['status'] == 'pending':
            amount = self.pre_trade_allocations[asset_id]['amount']
            self.pre_trade_allocations[asset_id]['status'] = 'expired_to_divestment'
            self.divestment += amount
            return True
        return False

    def pre_trade_to_held(self, asset_id):
        if asset_id in self.pre_trade_allocations and self.pre_trade_allocations[asset_id]['status'] == 'pending':
            amount = self.pre_trade_allocations[asset_id]['amount']
            self.pre_trade_allocations[asset_id]['status'] = 'expired_to_held'
            self.held_for_reinvestment += amount
            return True
        return False

    def held_to_pre_trade(self, amount, asset_id):
        if self.held_for_reinvestment >= amount or math.isclose(self.held_for_reinvestment, amount):
            self.held_for_reinvestment -= amount
            self.pre_trade_allocations[asset_id] = {'amount': amount, 'status': 'pending'}
            return True
        return False

    def held_to_main(self):
        if self.held_for_reinvestment > 0:
            self.main += self.held_for_reinvestment
            self.held_for_reinvestment = 0.0

    def reinvest_from_divestment(self, amount, asset_id):
        if self.divestment >= amount or math.isclose(self.divestment, amount):
            self.divestment -= amount
            # If the asset already has an active allocation, add to it. Otherwise, create a new one.
            if asset_id in self.active_trade_allocations and self.active_trade_allocations[asset_id][
                'status'] == 'active':
                self.active_trade_allocations[asset_id]['amount'] += amount
            else:
                self.active_trade_allocations[asset_id] = {'amount': amount, 'status': 'active'}
            return True
        return False

    def add_profit_to_main(self, amount, asset_id):
        if asset_id in self.active_trade_allocations and self.active_trade_allocations[asset_id]['status'] == 'active':
            self.main += amount
            self.active_trade_allocations[asset_id]['status'] = 'sold_for_profit'
            return True
        return False

    def add_divestment_to_pool(self, amount, asset_id):
        if asset_id in self.active_trade_allocations and self.active_trade_allocations[asset_id]['status'] == 'active':
            self.divestment += amount
            self.active_trade_allocations[asset_id]['status'] = 'sold_for_loss'
            return True
        return False

    def consolidate(self):
        active_pre_trade_capital = sum(
            info['amount'] for info in self.pre_trade_allocations.values() if info['status'] == 'pending')
        active_trade_capital = sum(
            info['amount'] for info in self.active_trade_allocations.values() if info['status'] == 'active')

        self.main += active_pre_trade_capital + active_trade_capital + self.divestment + self.held_for_reinvestment

        self.pre_trade_allocations.clear()
        self.active_trade_allocations.clear()
        self.divestment = 0.0
        self.held_for_reinvestment = 0.0

    def get_report(self):
        """Generates a DataFrame report of where all capital is currently held."""
        report = []
        for asset_id, info in self.pre_trade_allocations.items():
            report.append(
                {'pool': 'pre_trade', 'asset_id': asset_id, 'capital': info['amount'], 'status': info['status']})
        for asset_id, info in self.active_trade_allocations.items():
            report.append(
                {'pool': 'active_trade', 'asset_id': asset_id, 'capital': info['amount'], 'status': info['status']})

        if self.main > 1e-9:
            report.append({'pool': 'main', 'asset_id': 'unallocated', 'capital': self.main, 'status': 'available'})
        if self.divestment > 1e-9:
            report.append({'pool': 'divestment', 'asset_id': 'unallocated', 'capital': self.divestment,
                           'status': 'pending_reinvestment'})
        if self.held_for_reinvestment > 1e-9:
            report.append(
                {'pool': 'held_for_reinvestment', 'asset_id': 'unallocated', 'capital': self.held_for_reinvestment,
                 'status': 'held_for_next_cycle'})

        return pd.DataFrame(report)


class LiveTrader:
    """
    The main trading class that manages the overall budget, assets, and strategies.
    """

    def __init__(self, initial_budget, api_executor: ApiExecutor, dry_run_repeats_on_loss=False, capital_allocation_percent=0.996, reinvest_expired_immediately=True,
                 sell_at_trigger_price=False, dry_run_loss_threshold=1, dry_run_to_live_threshold=1.08,
                 convert_all_on_trigger=True, dry_run_max_count=1):
        self.api_executor = api_executor # Store the ApiExecutor instance
        self.assets = {}
        self.latest_prices = {}
        self.initial_prediction_count = 0
        self.cycle_capital = 0.0
        self.logger = TradeLogger()
        self.capital_allocation_percent = capital_allocation_percent

        self.live_budget = Budget(initial_budget)
        self.paper_budget = Budget()

        self.dry_run_mode = False
        self.activate_dry_run_next_cycle = False
        self.dry_run_repeats_on_loss = dry_run_repeats_on_loss
        self.initialized = False

        self.reinvest_expired_immediately = reinvest_expired_immediately
        self.num_expired_last_cycle = 0

        self.cycle_start_time = None
        self.MAX_CYCLE_DURATION = timedelta(days=7)
        self.REINVESTMENT_WINDOW = timedelta(days=4)

        self.last_price_updated = None

        self.sell_at_trigger_price = sell_at_trigger_price

        self.consecutive_cycle_losses = 0
        self.consecutive_cycle_gains = 0

        self.dry_run_loss_threshold = dry_run_loss_threshold
        self.dry_run_to_live_threshold = dry_run_to_live_threshold
        self.convert_all_on_trigger = convert_all_on_trigger

        self.dry_run_max_count = dry_run_max_count
        self.dry_run_count = 0

    @property
    def current_budget(self):
        """Returns the appropriate budget object based on the current mode."""
        return self.paper_budget if self.dry_run_mode else self.live_budget

    def can_process_predictions(self):
        return len(self.assets) == 0

    def process_predictions(self, coins_to_trade, current_time=None):
        now = current_time or datetime.now()

        if self.current_budget.held_for_reinvestment > 0 and self.num_expired_last_cycle > 0:
            self._reinvest_expired_capital(coins_to_trade, current_time=now)

        if self.assets:
            return

        if not self.initialized:
            self.logger.log_event(now, 'INITIALIZATION', {'initial_budget': self.live_budget.get_total()})
            self.initialized = True

        if self.activate_dry_run_next_cycle:
            self.dry_run_mode = True
            self.activate_dry_run_next_cycle = False
            self.logger.log_event(now, 'DRY_RUN_ACTIVATED',
                                  {'reason': f'{self.dry_run_loss_threshold}_consecutive_losses'})
        else:
            if self.dry_run_max_count >= self.dry_run_count:
                self.dry_run_mode = False
                self.dry_run_count = 0

        self.live_budget.consolidate()
        self.cycle_capital = self.live_budget.get_total()
        self.initial_prediction_count = len(coins_to_trade)
        self.cycle_start_time = now

        if self.dry_run_mode:
            self.paper_budget = Budget(self.cycle_capital)

        self.logger.log_event(now, 'PREDICTION', {
            'predicted_coins': coins_to_trade,
            'cycle_capital': self.cycle_capital,
            'is_dry_run': self.dry_run_mode
        })

        if not coins_to_trade:
            return

        total_budget_per_trade = self.cycle_capital / self.initial_prediction_count if self.initial_prediction_count > 0 else 0
        budget_per_trade = total_budget_per_trade * self.capital_allocation_percent

        for coin in coins_to_trade:
            if coin not in self.assets:
                if self.current_budget.allocate_to_pre_trade(budget_per_trade, coin):
                    initial_price = self.latest_prices.get(coin, 40000)
                    self.assets[coin] = {
                        "state": "pre-trade",
                        "strategy": PreTrade(initial_price, start_time=now),
                        "allocated_capital": budget_per_trade
                    }
                else:
                    self.logger.log_event(now, 'PRE_TRADE_FAIL', {'reason': 'insufficient_budget', 'asset_id': coin})

    def receive_next_price_tick(self, price_tick, current_time=None):
        now = current_time or datetime.now()
        self.latest_prices[price_tick.asset_id] = price_tick.price
        self.last_price_updated = now

        if self.dry_run_mode:
            self._check_for_dry_run_conversion(current_time=now)

        if self.cycle_start_time and now > (self.cycle_start_time + self.MAX_CYCLE_DURATION):
            self._force_liquidate_cycle(current_time=now)
            return

        self._check_for_individual_expirations(current_time=now)

        asset_id = price_tick.asset_id
        if asset_id in self.assets:
            asset_info = self.assets[asset_id]
            action, trigger_price = asset_info["strategy"].receive_next_price_tick(price_tick, current_time=now)

            if action == "buy":
                self.buy(asset_id, price_tick.price, current_time=now)
            elif action in ["sell_profit", "sell_divest"]:
                sale_price = trigger_price if self.sell_at_trigger_price else price_tick.price
                self.sell(asset_id, sale_price, reason=action, current_time=now)


    def last_price_tick_submitted(self, current_time=None):
        now = current_time or datetime.now()
        self._process_pending_reinvestments(current_time=now)

        if 800 < self.live_budget.get_total() < 900:
            self.dry_run_max_count = 2
        elif 700 < self.live_budget.get_total() < 799:
            self.dry_run_max_count = 3
        elif 600 < self.live_budget.get_total() < 699:
            self.dry_run_max_count = 4
        else:
            self.dry_run_max_count = 1

    def _check_for_individual_expirations(self, current_time=None):
        now = current_time or datetime.now()

        expired_assets = {
            asset_id: info for asset_id, info in self.assets.items()
            if info["state"] == "pre-trade" and info["strategy"].is_expired(current_time=now)
        }

        if not expired_assets: return

        for asset_id, info in expired_assets.items():
            allocated_capital = info['allocated_capital']

            if self.reinvest_expired_immediately:
                self.current_budget.pre_trade_to_divestment(asset_id)
            else:
                self.current_budget.pre_trade_to_held(asset_id)
                if not self.dry_run_mode:
                    self.num_expired_last_cycle += 1

            self.logger.log_event(now, 'EXPIRATION', {
                'asset_id': asset_id,
                'freed_capital': allocated_capital,
                'is_dry_run': self.dry_run_mode,
                'reinvest_immediately': self.reinvest_expired_immediately
            })
            del self.assets[asset_id]

        self._check_and_reset_if_all_trades_closed(current_time=now)

    def _reinvest_expired_capital(self, new_coins, current_time):
        if self.cycle_start_time and current_time > (self.cycle_start_time + self.REINVESTMENT_WINDOW):
            self.logger.log_event(current_time, 'REINVESTMENT_SKIPPED', {
                'reason': 'reinvestment_window_expired',
                'held_capital': self.current_budget.held_for_reinvestment
            })
            self.current_budget.held_to_main()
            self.num_expired_last_cycle = 0
            return

        if not new_coins:
            self.current_budget.held_to_main()
            self.num_expired_last_cycle = 0
            return

        coins_to_invest = random.sample(new_coins, k=min(len(new_coins), self.num_expired_last_cycle))
        capital_per_trade = self.current_budget.held_for_reinvestment / len(coins_to_invest)

        for coin in coins_to_invest:
            if self.current_budget.held_to_pre_trade(capital_per_trade, coin):
                initial_price = self.latest_prices.get(coin, 40000)
                self.assets[coin] = {
                    "state": "pre-trade",
                    "strategy": PreTrade(initial_price, start_time=current_time),
                    "allocated_capital": capital_per_trade
                }
                self.logger.log_event(current_time, 'PRE_TRADE_FROM_EXPIRED', {
                    'asset_id': coin,
                    'cost': capital_per_trade,
                    'is_dry_run': self.dry_run_mode
                })
                if coin in new_coins: new_coins.remove(coin)

        self.num_expired_last_cycle = 0

    def _process_pending_reinvestments(self, current_time=None):
        if math.isclose(self.current_budget.divestment, 0.0): return

        eligible_trades = {
            asset_id: info for asset_id, info in self.assets.items()
            if info["state"] == "trade" and 1.005 <= (
                        self.latest_prices.get(asset_id, 0) / info["strategy"].purchase_price) <= 1.1
        }

        if not eligible_trades: return

        now = current_time or datetime.now()
        capital_to_reinvest = self.current_budget.divestment
        reinvestment_per_trade = capital_to_reinvest / len(eligible_trades)

        for asset_id, info in eligible_trades.items():
            current_price = self.latest_prices.get(asset_id)
            trade_result = self.api_executor.execute_trade(asset_id, 'buy', reinvestment_per_trade, current_price)

            if trade_result["success"]:
                if self.current_budget.reinvest_from_divestment(reinvestment_per_trade, asset_id):
                    info["amount"] += trade_result["amount_executed"]
                    info["investment"] += reinvestment_per_trade

                self.logger.log_event(now, 'REINVESTMENT', {
                    'asset_id': asset_id,
                    'capital_reinvested': reinvestment_per_trade,
                    'is_dry_run': self.dry_run_mode
                })

        self.current_budget.divestment = 0.0

    def _force_liquidate_cycle(self, current_time):
        """Sells all active trades and expires all pre-trades if the cycle exceeds its max duration."""
        self.logger.log_event(current_time, 'FORCE_LIQUIDATE', {'reason': 'max_cycle_duration_exceeded'})

        for asset_id, info in list(self.assets.items()):
            if info['state'] == 'trade':
                price = self.latest_prices.get(asset_id)
                if price:
                    self.sell(asset_id, price, reason='force_liquidate', current_time=current_time)
            elif info['state'] == 'pre-trade':
                self.current_budget.pre_trade_to_divestment(asset_id)
                self.logger.log_event(current_time, 'EXPIRATION', {
                    'asset_id': asset_id, 'freed_capital': info['allocated_capital'], 'is_dry_run': self.dry_run_mode,
                    'reason': 'force_liquidate'
                })
                del self.assets[asset_id]

        self._check_and_reset_if_all_trades_closed(current_time)

    def _check_and_reset_if_all_trades_closed(self, current_time=None):
        if not self.assets:
            now = current_time or datetime.now()

            final_capital = self.current_budget.get_total()
            profit = final_capital - self.cycle_capital

            if self.dry_run_mode:
                self.dry_run_count += 1

            if profit < 0:
                self.consecutive_cycle_losses += 1
                self.consecutive_cycle_gains = 0
            else:
                self.consecutive_cycle_losses = 0
                self.consecutive_cycle_gains += 1
                self.dry_run_count = 0

            if self.consecutive_cycle_losses >= self.dry_run_loss_threshold:
                if not self.dry_run_mode or self.dry_run_repeats_on_loss:
                    self.activate_dry_run_next_cycle = True

                self.consecutive_cycle_losses = 0

            self.logger.log_event(now, 'CYCLE_END', {
                'profit': profit,
                'final_capital': final_capital,
                'next_cycle_dry_run': self.activate_dry_run_next_cycle,
                'was_dry_run': self.dry_run_mode
            })

            self.live_budget.consolidate()
            self.cycle_start_time = None

    def buy(self, asset_id, price, current_time=None):
        if asset_id not in self.assets or self.assets[asset_id]['state'] != 'pre-trade':
            return

        asset_info = self.assets[asset_id]
        budget_per_trade = asset_info['allocated_capital']

        trade_result = self.api_executor.execute_trade(asset_id, 'buy', budget_per_trade, price)

        if trade_result["success"]:
            if self.current_budget.pre_trade_to_active(asset_id):
                self.assets[asset_id] = {
                    "state": "trade",
                    "strategy": ActiveTrade(trade_result["execution_price"], asset_id),
                    "amount": trade_result["amount_executed"],
                    "investment": budget_per_trade
                }
                self.logger.log_event(current_time or datetime.now(), 'BUY', {
                    'asset_id': asset_id, 'amount': trade_result['amount_executed'],
                    'price': trade_result['execution_price'], 'cost': budget_per_trade,
                    'fee': trade_result['fee'], 'is_dry_run': self.dry_run_mode
                })

    def sell(self, asset_id, price, reason, current_time=None):
        if asset_id not in self.assets or self.assets[asset_id]['state'] != 'trade': return

        asset_info = self.assets[asset_id]
        amount_to_sell = asset_info["amount"]

        trade_result = self.api_executor.execute_trade(asset_id, 'sell', amount_to_sell, price)

        if trade_result["success"]:
            net_proceeds = trade_result["net_proceeds"]
            profit = net_proceeds - asset_info["investment"]

            self.logger.log_event(current_time or datetime.now(), 'SELL', {
                'asset_id': asset_id, 'reason': reason, 'amount': amount_to_sell,
                'price': trade_result['execution_price'], 'net_proceeds': net_proceeds,
                'profit': profit, 'fee': trade_result['fee'], 'is_dry_run': self.dry_run_mode
            })

            del self.assets[asset_id]

            if reason == "sell_divest" or reason == "force_liquidate":
                self.current_budget.add_divestment_to_pool(net_proceeds, asset_id)
            else:
                self.current_budget.add_profit_to_main(net_proceeds, asset_id)

            self._check_and_reset_if_all_trades_closed(current_time=current_time)

    def _check_for_dry_run_conversion(self, current_time):
        """
        If in dry-run mode, checks if any paper trade has become profitable
        enough to convert the whole cycle to a live one.
        """
        if not self.dry_run_mode:
            return

        for asset_id, info in self.assets.items():
            if info['state'] == 'trade':
                purchase_price = info['strategy'].purchase_price
                current_price = self.latest_prices.get(asset_id)

                if current_price and (current_price / purchase_price) > self.dry_run_to_live_threshold:
                    self._convert_dry_run_to_live(current_time, asset_id)
                    break

    def _convert_dry_run_to_live(self, current_time, triggering_asset_id):
        """
        Converts an in-progress dry-run cycle to a live trading cycle by
        committing real capital to the current paper positions.
        """
        self.logger.log_event(current_time, 'DRY_RUN_CONVERTED_TO_LIVE', {
            'reason': 'profit_threshold_met',
            'triggering_asset': triggering_asset_id,
            'conversion_mode': 'all' if self.convert_all_on_trigger else 'trigger_only'
        })

        self.dry_run_mode = False

        if self.convert_all_on_trigger:
            # Option 1: Convert all paper positions to live pre-trades
            # First, calculate the total live capital available and how to split it.
            self.live_budget.consolidate()
            capital_to_allocate = self.live_budget.get_total()
            num_assets_to_convert = len(self.paper_budget.pre_trade_allocations) + len(
                self.paper_budget.active_trade_allocations)
            capital_per_asset = capital_to_allocate / num_assets_to_convert if num_assets_to_convert > 0 else 0

            for asset_id, info in list(self.paper_budget.pre_trade_allocations.items()):
                if info['status'] == 'pending' and self.live_budget.allocate_to_pre_trade(capital_per_asset, asset_id):
                    self.assets[asset_id]['allocated_capital'] = capital_per_asset  # Update the asset's own record

            for asset_id, info in list(self.paper_budget.active_trade_allocations.items()):
                if info['status'] == 'active':
                    current_price = self.latest_prices.get(asset_id)
                    if current_price and self.live_budget.allocate_to_pre_trade(capital_per_asset, asset_id):
                        self.assets[asset_id]['strategy'] = PreTrade(current_price, start_time=current_time)
                        self.assets[asset_id]['state'] = 'pre-trade'
                        self.assets[asset_id]['allocated_capital'] = capital_per_asset
        else:
            # Option 2: Consolidate all available live capital into a single pre-trade for the triggering asset
            capital_to_allocate = self.live_budget.get_total()
            self.live_budget.consolidate()

            self.assets = {asset_id: info for asset_id, info in self.assets.items() if asset_id == triggering_asset_id}

            current_price = self.latest_prices.get(triggering_asset_id)
            if current_price and self.live_budget.allocate_to_pre_trade(capital_to_allocate, triggering_asset_id):
                self.assets[triggering_asset_id]['state'] = 'pre-trade'
                self.assets[triggering_asset_id]['strategy'] = PreTrade(current_price, start_time=current_time)
                self.assets[triggering_asset_id]['allocated_capital'] = capital_to_allocate

        self.paper_budget = Budget()
