import pandas as pd


def heikin_ashi(df_src, smooth=1):
    # calculate HeikinAshi candelsticks
    # return DataFrame("Open", "High", "Low", "Close")
    """
    - Calculate HeikinAshi(HA) candelstick
        - Formula of HA
            1. The Heikin-Ashi Close is simply an average of the open,
            high, low and close for the current period.

                HA-Close = (Open(0) + High(0) + Low(0) + Close(0)) / 4

            2. The Heikin-Ashi Open is the average of the prior Heikin-Ashi
            candlestick open plus the close of the prior Heikin-Ashi candlestick.

                HA-Open = (HA-Open(-1) + HA-Close(-1)) / 2

            3. The Heikin-Ashi High is the maximum of three data points:
            the current period's high, the current Heikin-Ashi
            candlestick open or the current Heikin-Ashi candlestick close.

                HA-High = Maximum of the High(0), HA-Open(0) or HA-Close(0)

            4. The Heikin-Ashi low is the minimum of three data points:
            the current period's low, the current Heikin-Ashi
            candlestick open or the current Heikin-Ashi candlestick close.

                HA-Low = Minimum of the Low(0), HA-Open(0) or HA-Close(0)
        - Ref: https://school.stockcharts.com/doku.php?id=chart_analysis:heikin_ashi
        - Custom smoothed version:
            HA-Open = Average of smooth HA-Open previous candelsticks(origin smooth=1)
    """
    ha = []
    for i, row in df_src.iterrows():
        if i < smooth:
            ha.append(
                (
                    (row["Open"] + row["Close"]) / 2,  # HA open
                    row["High"],  # HA high
                    row["Low"],  # HA low
                    (row["Open"] + row["High"] + row["Low"] + row["Close"]) / 4,  # HA close
                )
            )
        else:
            prev_ha = ha[i - smooth :]
            ha_close = (row["Open"] + row["High"] + row["Low"] + row["Close"]) / 4
            ha_open = sum([(ph[0] + ph[3]) / 2 for ph in prev_ha]) / len(prev_ha)
            ha_high = max(row["High"], ha_open, ha_close)
            ha_low = min(row["Low"], ha_open, ha_close)
            ha.append((ha_open, ha_high, ha_low, ha_close))
    return pd.DataFrame(ha, columns=["Open", "High", "Low", "Close"])

def heikin_ashi_stream(ha_df, last_kline, smooth=1):
    # return HA of last_kline
    prev_ha = ha_df[-smooth :]
    ha_close = (last_kline["Open"] + last_kline["High"] + last_kline["Low"] + last_kline["Close"]) / 4
    ha_open = sum([(ph["Open"] + ph["Close"]) / 2 for _, ph in prev_ha.iterrows()]) / len(prev_ha)
    ha_high = max(last_kline["High"], ha_open, ha_close)
    ha_low = min(last_kline["Low"], ha_open, ha_close)
    return pd.DataFrame([(ha_open, ha_high, ha_low, ha_close)], columns=["Open", "High", "Low", "Close"])