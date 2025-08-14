import pickle
import time

import requests
import pandas as pd
from datetime import datetime, timedelta


# --- Helper Functions for Time Conversion (Updated) ---

def convert_to_local_datetime(dt_str, local_offset_hours=0):
    """Converts a datetime string to local time with an offset."""
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    dt_local = dt + timedelta(hours=local_offset_hours)  # Adjust backwards to match UTC time
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')


def convert_timestamp_to_local(ts, local_offset_hours=2):
    """Converts a UNIX timestamp (milliseconds) to local time."""
    dt_utc = datetime.utcfromtimestamp(ts / 1000)  # Convert to UTC datetime
    dt_local = dt_utc + timedelta(hours=local_offset_hours)  # Apply offset
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

def load_transformed_coins_pickle(file_path="./assets/transformed_coins.pkl"):
    """Loads transformed coins from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            transformed_coins = pickle.load(f)
        print(f"Loaded transformed coins from {file_path}")
        return transformed_coins
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

# --- Main DataStream Class ---

class PriceTick:
    """Represents a single price point for an asset at a specific time."""

    def __init__(self, asset_id, price, timestamp, fear_greed_index=None):
        self.asset_id = asset_id
        self.price = price
        self.timestamp = timestamp
        self.fear_greed_index = fear_greed_index


    def __repr__(self):
        return f"PriceTick(asset='{self.asset_id}', price={self.price}, time='{self.timestamp}')"


class DataStream:
    """
    Manages fetching real price data from an API and notifying observers.
    """

    def __init__(self, asset_universe = []):
        print("Initializing DataStream with real API fetching...")
        self._observers = []
        self._asset_universe = asset_universe
        self._assets_to_track = []  # This will be a list of currency IDs

        transformed_coins = load_transformed_coins_pickle()
        self.slug_currency_id_map = {coin['slug'] : str(coin['currencyId']) for coin in transformed_coins}

    def subscribe(self, observer):
        """Adds an observer (e.g., a trading agent) to the notification list."""
        if observer not in self._observers:
            self._observers.append(observer)
            print(f"Observer {observer.__class__.__name__} subscribed.")

    def _notify_observers(self, price_tick: PriceTick):
        """Pushes a new price tick to all subscribed observers."""
        for observer in self._observers:
            # The trading agent expects a 'receive_next_price_tick' or 'update' method
            if hasattr(observer, 'receive_next_price_tick'):
                observer.receive_next_price_tick(price_tick)
            elif hasattr(observer, 'update'):
                observer.update(price_tick)

    def update_tracked_assets(self, new_asset_list: list[str]):
        """Updates the list of assets (currency IDs) to fetch prices for."""
        print(f"DataStream is now tracking: {new_asset_list}")
        self._assets_to_track = new_asset_list

    def _fetch_currency_historical(self, before_str, after_str, currency_ids, with_price_change=True, local_offset_hours=2):
        """Fetches historical currency data, converting timestamps to local time and adjusting input times."""

        # Convert input strings to local-adjusted UTC time before converting to UNIX timestamp

        # Convert only if not already datetime
        before_local = convert_to_local_datetime(before_str)

        after_local = convert_to_local_datetime(after_str)

        # Convert local-adjusted UTC time to UNIX timestamps (milliseconds)
        before = int(datetime.strptime(before_local, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
        after = int(datetime.strptime(after_local, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

        url = "https://api2.icodrops.com/portfolio/api/currencyHistorical"
        params = {
            "before": before,
            "after": after,
            "currencyIds": currency_ids,
            "withPriceChange": with_price_change
        }

        response = requests.get(url, params=params, timeout=20)

        if response.status_code == 200:
            data = response.json()

            # Convert timestamps
            timestamps = data.get("timestamps", [])
            readable_timestamps = [convert_timestamp_to_local(ts) for ts in timestamps]  # Convert to local time

            # Extract data for the specified currency ID (ensure it's a string key)
            currency_data = data.get("data", {}).get(str(currency_ids), {})

            # Convert data into a DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "local_time": readable_timestamps,  # Local timestamp column
                "priceUSD": [p.get("USD", None) for p in currency_data.get("prices", [])],
                "fdv": currency_data.get("fdv", []),
                "volume24h": currency_data.get("volumes24h", []),
                "marketCap": currency_data.get("marketCaps", []),
                "fearGreedIndex": currency_data.get("fearGreedIndexes", []),
                "priceChange": currency_data.get("priceChanges", []),
                "funding": currency_data.get("fundings", [])
            })
            df['local_time'] = pd.to_datetime(df['local_time'])

            return df
        else:
            print(f"Error fetching data: {response.status_code}")
            raise Exception(f"Error fetching data: {response.status_code}")

    def _fetch_all_historical_data(self, currency_ids, slug, before_time, after_time, limit_date=datetime.strptime("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
                                  return_df=False):
        """Fetches all historical data for a given currency_id, from today to the earliest available date, using a dynamically computed interval."""

        # Convert before/after time strings to datetime objects
        before_dt = datetime.strptime(before_time, "%Y-%m-%d %H:%M:%S")
        after_dt = datetime.strptime(after_time, "%Y-%m-%d %H:%M:%S")

        # Calculate interval dynamically in hours
        interval_hours = (before_dt - after_dt).total_seconds() / 3600

        end_time = datetime.now()  # Today's date in UTC
        start_time = end_time - timedelta(hours=interval_hours)  # Step backward using computed interval

        all_data = []  # Store collected data

        pages_counted = 0

        no_older_data_left = True

        while no_older_data_left:
            # Convert time ranges to formatted strings
            before_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            after_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

            if datetime.strptime(before_str, "%Y-%m-%d %H:%M:%S") < limit_date:
                no_older_data_left = False
                break  # Stop fetching if we've reached the limit

            max_retries = 10
            retries = 0
            base_delay = 5
            while retries < max_retries:
                try:
                    # Fetch historical data for this time window
                    df = self._fetch_currency_historical(before_str, after_str, currency_ids)

                    if df is None or df.empty:
                        no_older_data_left = False
                        break  # Stop if there’s no older data left

                    # Append fetched data
                    all_data.append(df)

                    # Move further back in time, ensuring at least a **3-hour overlap** on both ends
                    end_time = start_time + timedelta(hours=5)  # Ensure overlap with previous batch
                    start_time = end_time - timedelta(hours=interval_hours - 5)  # Step back while maintaining overlap

                    # print(f"Interval: {end_time} <=> {start_time}")

                    pages_counted += 1

                    if len(df) < 650:
                        df
                    # sys.stdout.write(f"\rPages counted so far: {pages_counted}. Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}            ")
                    # sys.stdout.flush()
                    break
                except Exception as e:
                    retries += 1
                    # delay = base_delay * (2 ** retries)  # Exponential backoff
                    delay = base_delay  # Exponential backoff
                    # print(f"Error for {slug}: {e}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                if retries == max_retries:
                    print(f"Max retries reached for {slug}. Moving to next iteration.")
                    break  # Stop fetching if max retries are exceeded
            if retries == max_retries:
                print(f"Max retries reached for {slug}. Ending iteration.")
                break  # Stop fetching if max retries are exceeded

        result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=['timestamp']) if all_data else None
        result.sort_values(by=['timestamp'], ascending=True, inplace=True)

        return result

    def fetch_and_notify(self):
        """
        Fetches the latest price for each tracked asset and notifies observers.
        This is designed to be called periodically by the runner (e.g., every 5 mins).
        """
        if not self._assets_to_track:
            return

        print(f"Fetching latest ticks at {datetime.now().strftime('%H:%M:%S')}...")
        now = datetime.now()
        # Fetch data from the last 10 minutes to ensure we get the most recent tick
        ten_minutes_ago = now - timedelta(minutes=10)

        for asset_id in self._assets_to_track:

            currency_id = self.slug_currency_id_map[asset_id]

            max_retries = 10
            retries = 0
            base_delay = 5
            while retries < max_retries:
                try:
                    df = self._fetch_currency_historical(now.strftime('%Y-%m-%d %H:%M:%S'), ten_minutes_ago.strftime('%Y-%m-%d %H:%M:%S'), currency_id)

                    if df is not None and not df.empty:
                        # The last row in the DataFrame is the most recent tick
                        latest_tick_data = df.iloc[-1]

                        price_tick = PriceTick(
                            asset_id=asset_id,
                            price=latest_tick_data['priceUSD'],
                            timestamp=datetime.fromtimestamp(latest_tick_data['timestamp'] / 1000),
                            fear_greed_index=latest_tick_data.get('fearGreedIndex')
                        )

                        print(f"  > Notifying for: {price_tick}")
                        self._notify_observers(price_tick)
                        break
                except Exception as e:
                    retries += 1
                    delay = base_delay  # Exponential backoff
                    time.sleep(delay)

                if retries == max_retries:
                    print(f"Max retries reached for {asset_id}. Moving to next iteration.")
                    break  # Stop fetching if max retries are exceeded

    def get_historical_prices(self, earliest_date: datetime) -> dict[str, pd.DataFrame]:
        """
        Performs an on-demand fetch for a history of prices for all tracked assets.
        """
        historical_data = {asset_id: pd.DataFrame.empty for asset_id in self._assets_to_track}
        now = datetime.utcnow()

        if earliest_date >= now:
            return historical_data

        for asset_id in self._asset_universe:
            print(f"Fetching historical data for {asset_id} from {earliest_date} to {now}...")

            before_time = "2025-05-17 01:00:00"
            after_time = "2025-05-13 20:00:00"

            currency_id = self.slug_currency_id_map[asset_id]
            df = self._fetch_all_historical_data(currency_id, asset_id, before_time, after_time, earliest_date)
            historical_data[asset_id] = df

        return historical_data
