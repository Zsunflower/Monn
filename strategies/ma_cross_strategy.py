import logging
import pandas as pd
from datetime import datetime
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus

bot_logger = logging.getLogger("bot_logger")
IN_BUYING = 1
IN_SELLING = 2


class MACross(BaseStrategy):
    """
    Test broker API strategy
    random place order and close
    use only for testing API
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
        self.fast_ma = self.ma_func(self.tfs_chart[self.tf]["Close"], self.params["fast_ma"])
        self.slow_ma = self.ma_func(self.tfs_chart[self.tf]["Close"], self.params["slow_ma"])
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        last_kline = self.tfs_chart[self.tf].iloc[-1]  # last kline
        self.fast_ma.loc[len(self.fast_ma)] = self.ma_stream_func(
            self.tfs_chart[self.tf]["Close"], self.params["fast_ma"]
        )
        self.slow_ma.loc[len(self.slow_ma)] = self.ma_stream_func(
            self.tfs_chart[self.tf]["Close"], self.params["slow_ma"]
        )

    def check_required_params(self):
        return all([key in self.params.keys() for key in ["fast_ma", "slow_ma", "type"]])

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return self.params["fast_ma"] < self.params["slow_ma"] and self.params["type"] in ["SMA", "EMA"]

    def update(self, tf):
        # update when new kline arrive
        super().update(tf)
        # determind trend
        if len(self.orders_opening) == 0:
            self.state = None
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        if self.state is None:
            if (
                self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1]
                and self.slow_ma.iloc[-1] > ta.stream.SMA(chart["Close"], 100)
                and last_kline["Close"] >= ta.stream.SMA(chart["Close"], 100)
                and last_kline["Close"] > last_kline["Open"]
            ):
                # new buy order
                order = Order(OrderType.MARKET, OrderSide.BUY, last_kline["Close"], status=OrderStatus.FILLED)
                order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                if order:
                    if order.type == OrderType.MARKET:
                        order["FILL_TIME"] = last_kline["Open time"]
                    order["strategy"] = self.name
                    order["description"] = self.description
                    order["desc"] = {}
                    self.trader.create_trade(order, self.volume)
                    self.orders_opening.append(order)
                    self.state = IN_BUYING
            elif (
                self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1]
                and self.slow_ma.iloc[-1] < ta.stream.SMA(chart["Close"], 100)
                and last_kline["Close"] <= ta.stream.SMA(chart["Close"], 100)
                and last_kline["Close"] < last_kline["Open"]
            ):
                # new sell order
                order = Order(OrderType.MARKET, OrderSide.SELL, last_kline["Close"], status=OrderStatus.FILLED)
                order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                if order:
                    if order.type == OrderType.MARKET:
                        order["FILL_TIME"] = last_kline["Open time"]
                    order["strategy"] = self.name
                    order["description"] = self.description
                    order["desc"] = {}
                    self.trader.create_trade(order, self.volume)
                    self.orders_opening.append(order)
                    self.state = IN_SELLING
        elif self.state == IN_BUYING and last_kline["Close"] < last_kline["Open"]:
            if self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1]:
                # close buy order
                self.close_order_by_side(last_kline, OrderSide.BUY)
                self.state = None
        elif self.state == IN_SELLING and last_kline["Close"] > last_kline["Open"]:
            if self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1]:
                # close sell order
                self.close_order_by_side(last_kline, OrderSide.SELL)
                self.state = None

    def plot_orders(self):
        fig = make_subplots(2, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.8, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            yaxis=dict(showgrid=False),
            xaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            plot_bgcolor="#2E4053",
            paper_bgcolor="#797D7F",
            font=dict(color="#F7F9F9"),
        )
        df = self.tfs_chart[self.tf]
        dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
        tmp_ot = df["Open time"]
        df["Open time"] = list(range(len(df)))
        super().plot_orders(fig, self.tf, 1, 1, dt2idx=dt2idx)
        df = self.tfs_chart[self.tf]
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
            go.Scatter(
                x=df["Open time"], y=ta.SMA(df["Close"], 100), mode="lines", line=dict(color="orange"), name="MA_100"
            ),
            row=1,
            col=1,
        )

        colors = ["red" if kline["Open"] > kline["Close"] else "green" for i, kline in df.iterrows()]
        fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], marker_color=colors), row=2, col=1)

        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
        fig.update_layout(
            title={"text": "MA Cross Strategy (tf {})".format(self.tf), "x": 0.5, "xanchor": "center"}
        )
        df["Open time"] = tmp_ot
        return fig
