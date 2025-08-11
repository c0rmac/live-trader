import pandas as pd
from datetime import datetime


class TradeLogger:
    """
    A class dedicated to recording all trading activities and events.
    It can also process the log to generate summary tables.
    """

    def __init__(self):
        self.log = []

    def log_event(self, timestamp, event_type, details):
        """
        A generic method to log various events.

        :param timestamp: The time of the event.
        :param event_type: A string describing the event (e.g., 'PREDICTION', 'EXPIRATION').
        :param details: A dictionary containing event-specific information.
        """
        self.log.append({
            "timestamp": timestamp,
            "event_type": event_type,
            **details
        })

    def get_log_df(self):
        """
        Returns the entire log as a pandas DataFrame for analysis.
        """
        if not self.log:
            return pd.DataFrame()

        df = pd.DataFrame(self.log)
        # Reorder columns for better readability
        cols = ['timestamp', 'event_type'] + [col for col in df.columns if col not in ['timestamp', 'event_type']]
        return df[cols]

    def get_profit_table(self):
        """
        Processes the log to create a summary table of profits and trades per asset.
        """
        log_df = self.get_log_df()
        if log_df.empty or 'SELL' not in log_df['event_type'].unique():
            return pd.DataFrame(columns=['Total Profit', 'Num Trades', 'Wins', 'Losses', 'Win Rate (%)'])

        # Filter for actual sales events
        sell_events = log_df[log_df['event_type'] == 'SELL'].copy()

        if sell_events.empty:
            return pd.DataFrame(columns=['Total Profit', 'Num Trades', 'Wins', 'Losses', 'Win Rate (%)'])

        # Calculate wins and losses
        sell_events['is_win'] = (sell_events['profit'] > 0).astype(int)
        sell_events['is_loss'] = (sell_events['profit'] <= 0).astype(int)

        # Aggregate the results
        profit_table = sell_events.groupby('asset_id').agg(
            total_profit=('profit', 'sum'),
            num_trades=('event_type', 'count'),
            wins=('is_win', 'sum'),
            losses=('is_loss', 'sum')
        ).reset_index()

        # Calculate Win Rate
        profit_table['win_rate_%'] = (profit_table['wins'] / profit_table['num_trades'] * 100).round(2)

        # Rename columns for clarity
        profit_table.rename(columns={
            'asset_id': 'Asset',
            'total_profit': 'Total Profit',
            'num_trades': 'Num Trades',
            'wins': 'Wins',
            'losses': 'Losses',
            'win_rate_%': 'Win Rate (%)'
        }, inplace=True)

        return profit_table.set_index('Asset')

    def get_cycle_returns_table(self):
        """
        Processes the log to create a summary table of performance for each trading cycle.
        """
        log_df = self.get_log_df()
        if log_df.empty or 'CYCLE_END' not in log_df['event_type'].unique():
            return pd.DataFrame()

        # Filter for events that define cycle boundaries and conversions
        cycle_events = log_df[log_df['event_type'] == 'CYCLE_END'].copy()
        prediction_events = log_df[log_df['event_type'] == 'PREDICTION'].copy()
        conversion_events = log_df[log_df['event_type'] == 'DRY_RUN_CONVERTED_TO_LIVE'].copy()

        if cycle_events.empty:
            return pd.DataFrame()

        # Determine if a conversion happened within each cycle
        converted_in_cycle = []
        for index, cycle_end in cycle_events.iterrows():
            # Find the prediction that started this cycle
            cycle_start_event = prediction_events[prediction_events['timestamp'] < cycle_end['timestamp']].iloc[-1]
            start_time = cycle_start_event['timestamp']
            end_time = cycle_end['timestamp']

            # Check if any conversion event falls within this cycle's timeframe
            conversion_in_this_cycle = conversion_events[
                (conversion_events['timestamp'] >= start_time) &
                (conversion_events['timestamp'] <= end_time)
                ]
            converted_in_cycle.append(not conversion_in_this_cycle.empty)

        cycle_events['converted_to_live'] = converted_in_cycle

        # Calculate starting capital and return ratio
        cycle_events['start_capital'] = cycle_events['final_capital'] - cycle_events['profit']
        # Avoid division by zero if a cycle starts with no capital
        cycle_events['return_ratio_%'] = (cycle_events['profit'] / cycle_events['start_capital'] * 100).fillna(0).round(
            4)

        # Select and rename columns for the final table
        cycle_table = cycle_events[[
            'timestamp', 'start_capital', 'profit', 'final_capital', 'return_ratio_%', 'was_dry_run',
            'converted_to_live'
        ]].rename(columns={
            'timestamp': 'Cycle End Time',
            'start_capital': 'Start Capital',
            'profit': 'Profit',
            'final_capital': 'End Capital',
            'return_ratio_%': 'Return Ratio (%)',
            'was_dry_run': 'Was Dry Run',
            'converted_to_live': 'Converted to Live'
        })

        return cycle_table.reset_index(drop=True)
