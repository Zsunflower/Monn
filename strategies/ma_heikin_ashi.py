import logging
import pandas as pd
from datetime import datetime
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus
from utils import get_line_coffs, find_uptrend_line, find_downtrend_line, get_y_on_line


bot_logger = logging.getLogger("bot_logger")
UP_TREND = 0
DOWN_TREND = 1
IN_BUYING = 0
IN_SELLING = 1


class MAHeikinAshi(BaseStrategy):
    """
    - EMA HeikinAshi strategy
    """

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.state = None
        self.trend = None
        self.ma_func = ta.SMA if self.params["type"] == "SMA" else ta.EMA
        self.ma_stream_func = ta.stream.SMA if self.params["type"] == "SMA" else ta.stream.EMA

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate HA candelstick
        chart = self.tfs_chart[self.tf]
        self.ha = mta.heikin_ashi(chart, self.params["ha_smooth"])
        self.fast_ma = self.ma_func(self.tfs_chart[self.tf]["Close"], self.params["fast_ma"])
        self.slow_ma = self.ma_func(self.tfs_chart[self.tf]["Close"], self.params["slow_ma"])
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        last_kline = self.tfs_chart[self.tf].iloc[-1]  # last kline
        self.ha = pd.concat([self.ha, mta.heikin_ashi_stream(self.ha, last_kline, self.params["ha_smooth"])])
        self.fast_ma.loc[len(self.fast_ma)] = self.ma_stream_func(
            self.tfs_chart[self.tf]["Close"], self.params["fast_ma"]
        )
        self.slow_ma.loc[len(self.slow_ma)] = self.ma_stream_func(
            self.tfs_chart[self.tf]["Close"], self.params["slow_ma"]
        )

    def check_required_params(self):
        return all([key in self.params.keys() for key in ["ha_smooth", "fast_ma", "slow_ma", "type", "n_kline_trend"]])

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return self.params["fast_ma"] < self.params["slow_ma"] and self.params["type"] in ["SMA", "EMA"]

    def update(self, tf):
        # update when new kline arrive
        super().update(tf)
        # determind trend
        chart = self.tfs_chart[self.tf]
        n_df = chart[-self.params["n_kline_trend"] :].reset_index()
        n_last_poke_points = []
        n_last_peak_points = []
        for i, kline in n_df.iterrows():
            n_last_poke_points.append((len(chart) - self.params["n_kline_trend"] + i, kline["Low"]))
            n_last_peak_points.append((len(chart) - self.params["n_kline_trend"] + i, kline["High"]))
        self.up_trend_line = find_uptrend_line(n_last_poke_points)
        self.down_trend_line = find_downtrend_line(n_last_peak_points)
        self.up_pct = (self.up_trend_line[1][1] - self.up_trend_line[0][1]) / self.up_trend_line[0][1]
        self.down_pct = (self.down_trend_line[1][1] - self.down_trend_line[0][1]) / self.down_trend_line[0][1]
        self.check_signal(self.tfs_chart[self.tf].iloc[-1])

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self, last_kline):
        last_ha = self.ha.iloc[-1]
        if self.state is None:
            if self.up_pct > 0.03 and self.down_pct > 0 and self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1]:
                if last_ha["Open"] < last_ha["Close"] and last_ha["Open"] == last_ha["Low"]:  # HA open < HA close
                    # new buy order
                    order = Order(OrderType.MARKET, OrderSide.BUY, last_kline["Close"], status=OrderStatus.FILLED)
                    order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                    if order:
                        if order.type == OrderType.MARKET:
                            order["FILL_TIME"] = last_kline["Open time"]
                        order["strategy"] = self.name
                        order["description"] = self.description
                        order["desc"] = {"up_trend_line": self.up_trend_line, "down_trend_line": self.down_trend_line}
                        self.trader.create_trade(order, self.volume)
                        self.orders_opening.append(order)
                        self.state = IN_BUYING
            elif self.down_pct < -0.03 and self.up_pct < 0 and self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1]:
                if last_ha["Open"] > last_ha["Close"] and last_ha["Open"] == last_ha["High"]:  # HA open > HA close
                    # new sell order
                    order = Order(OrderType.MARKET, OrderSide.SELL, last_kline["Close"], status=OrderStatus.FILLED)
                    order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                    if order:
                        if order.type == OrderType.MARKET:
                            order["FILL_TIME"] = last_kline["Open time"]
                        order["strategy"] = self.name
                        order["description"] = self.description
                        order["desc"] = {"up_trend_line": self.up_trend_line, "down_trend_line": self.down_trend_line}
                        self.trader.create_trade(order, self.volume)
                        self.orders_opening.append(order)
                        self.state = IN_SELLING
        elif self.state == IN_BUYING:
            if self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1] and self.down_pct < 0:
                # close buy order
                self.close_order_by_side(last_kline, OrderSide.BUY)
                self.state = None
        elif self.state == IN_SELLING:
            if self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1] and self.up_pct > 0:
                # close sell order
                self.close_order_by_side(last_kline, OrderSide.SELL)
                self.state = None

    def plot_orders(self):
        fig = make_subplots(3, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.4, 0.4, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            yaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            yaxis3=dict(showgrid=False),
            plot_bgcolor="#2E4053",
            paper_bgcolor="#797D7F",
            font=dict(color="#F7F9F9"),
        )
        df = self.tfs_chart[self.tf]
        dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
        tmp_ot = df["Open time"]
        df["Open time"] = list(range(len(df)))
        super().plot_orders(fig, self.tf, 1, 1)

        for order in self.orders_closed:
            desc = order["desc"]
            up_trend = desc["up_trend_line"]
            down_trend = desc["down_trend_line"]
            fig.add_shape(
                type="line",
                x0=df.iloc[up_trend[0][0]]["Open time"],
                y0=up_trend[0][1],
                x1=df.iloc[up_trend[1][0]]["Open time"],
                y1=up_trend[1][1],
                line=dict(color="green"),
                row=1,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=df.iloc[down_trend[0][0]]["Open time"],
                y0=down_trend[0][1],
                x1=df.iloc[down_trend[1][0]]["Open time"],
                y1=down_trend[1][1],
                line=dict(color="red"),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.fast_ma,
                mode="lines",
                line=dict(color="blue"),
                name="FastMA_{}".format(self.params["fast_ma"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.slow_ma,
                mode="lines",
                line=dict(color="red"),
                name="SlowMA_{}".format(self.params["slow_ma"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Candlestick(
                x=df["Open time"],
                open=self.ha["Open"],
                high=self.ha["High"],
                low=self.ha["Low"],
                close=self.ha["Close"],
            ),
            row=2,
            col=1,
        )
        colors = ["red" if kline["Open"] > kline["Close"] else "green" for i, kline in df.iterrows()]
        fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], marker_color=colors), row=3, col=1)

        fig.update_layout(title={"text": "Heikin AShi Strategy (tf {})".format(self.tf), "x": 0.5, "xanchor": "center"})
        df["Open time"] = tmp_ot
        return fig
