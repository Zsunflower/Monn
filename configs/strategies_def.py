[
    {
        "name": "ma_heikin_ashi",
        "params": {
            "ha_smooth": int, 
            "fast_ma": int, 
            "slow_ma": int, 
            "type": ["EMA", "SMA"], 
            "n_kline_trend": int
        }
    },
    {
        "name": "rsi_hidden_divergence",
        "params": {
            "rsi_len": int,  # RSI length, Ex 14
            "delta_rsi": int,  # delta for compare RSI, rsi_1 < rsi_2 <-> rsi_1 + delta_rsi < rsi_2 , Ex 3
            "delta_price_pct": float,  # price percent for compare price, price_1 < price_2 <-> price_2 > (1 + delta_price_pct / 100) * price_!, Ex 0.2
            "n_last_point": int,  # number of zz_points lookback to search for hidden divergence
            "min_rr": int,
            "min_rw_pct": float,
            "type": ["DIVER", "HIDDEN_DIVER", "BOTH"],
            "min_zz_pct": float,  # minimum of zigzag strength
            "min_trend_pct": float,  # minimum price percent to determine a trend, Ex 6%
            "min_updown_ratio": float,  # maximum close price follow up/down trend lines [0 - 1], Ex 0.5
            "zz_type": ["ZZ_CONV", "ZZ_DRC"],
            "zz_conv_size": int,
            "sl_fix_mode": ["IGNORE, ADJ_SL, ADJ_ENTRY"],
            "n_trend_point": int,  # number of zz_points to determine up/down trend
        }
    },
    {
        "name": "macd_divergence",
        "params": {
            "macd_inputs": {"fast_len": int, "slow_len": int, "signal": int},
            "macd_type": ["MACD", "MACD_SIGNAL", "MACD_HIST"],
            "delta_macd": int,
            "delta_price_pct": float,
            "n_last_point": int,
            "min_rr": float,
            "min_rw_pct": float,
            "type": ["DIVER", "HIDDEN_DIVER", "BOTH"],
            "min_zz_pct": float,
            "min_trend_pct": float,
            "min_updown_ratio": float,
            "zz_type": ["ZZ_CONV", "ZZ_DRC"],
            "zz_conv_size": int,
            "sl_fix_mode": ["IGNORE, ADJ_SL, ADJ_ENTRY"],
            "n_trend_point": int,
        }
    }
]
