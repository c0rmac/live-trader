import pandas as pd

def prepare_lookback(
        pre_xy: pd.DataFrame,
        fear_greed_trail: list,
        lookback_step: int,
        candle_interval: str
):

    window_size = 24
    if candle_interval == "1h":
        window_size = 24
    elif candle_interval == "1d":
        window_size = 1
    elif candle_interval == "12h":
        window_size = 2
    elif candle_interval == "4h":
        window_size = int(24/4)

    def rolling_mode(series):
        mode_values = series.mode()
        return mode_values.iloc[0] if not mode_values.empty else None  # Handles empty cases

    if lookback_step == 0:
        pre_xy["mode_fearGreedIndex_step0"] = pre_xy["fearGreedIndex"].shift(0).rolling(window=window_size).apply(lambda x: rolling_mode(x), raw=False)
    else:
        pre_xy[f"mode_fearGreedIndex_step{lookback_step}"] = (pre_xy["fearGreedIndex"]
                                                              .shift((window_size * lookback_step) + 1)
                                                              .rolling(window=window_size)
                                                              .apply(lambda x: rolling_mode(x), raw=False))

    #pre_xy[f"mode_fearGreedIndex_step{lookback_step}"] = pre_xy["fearGreedIndex"].shift(- (24 * lookback_step)).rolling(window=24).apply(lambda x: rolling_mode(x), raw=False)
    fear_greed_trail.append(f"mode_fearGreedIndex_step{lookback_step}")

    return pre_xy, fear_greed_trail

def generate_lookback(candle_df):
    max_lookback = 60

    fear_greed_trail = []

    for lookback in range(max_lookback + 1):
        candle_df, fear_greed_trail = prepare_lookback(candle_df, fear_greed_trail, lookback, "1d")

    return candle_df, fear_greed_trail