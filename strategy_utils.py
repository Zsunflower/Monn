from strategies import *

strategies = {
    "ma_cross": "MACross",
    "ma_heikin_ashi": "MAHeikinAshi",
    "rsi_hidden_divergence": "RSIDivergence",
    "macd_divergence": "MACDDivergence",
    "rsi_regular_divergence": "RSIRegularDivergence",
    "break_strategy": "BreakStrategy",
    "price_action": "PriceAction",
    "trend_following": "TrendFollowing",
    "zigzag_follower": "ZigzagFollower",
}


def load_strategy(strategy_def):
    strategy_name = strategy_def["name"]
    return globals()[strategies[strategy_name]](strategy_name, strategy_def["params"], strategy_def["tfs"])
