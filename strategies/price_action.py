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


class PriceAction(BaseStrategy):
    """
    - PriceAction strategy
    """

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.state = None
        self.trend = None
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        self.temp = []
        self.checked_pidx = []

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate HA candelstick
        chart = self.tfs_chart[self.tf]
        self.ma_vol = ta.SMA(chart["Volume"], self.params["ma_vol"])
        self.zz_points = mta.zigzag(chart, self.min_zz_ratio)
        self.init_main_zigzag()
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def init_main_zigzag(self):
        self.main_zz_idx = []
        if len(self.zz_points) > 0:
            self.main_zz_idx.append(0)
        else:
            return
        last_main_zz_idx = 0
        while last_main_zz_idx + 3 < len(self.zz_points):
            last_main_zz_type = self.zz_points[last_main_zz_idx].ptype
            if last_main_zz_type == mta.POINT_TYPE.PEAK_POINT:
                if (
                    self.zz_points[last_main_zz_idx + 2].pline.high < self.zz_points[last_main_zz_idx].pline.high
                    and self.zz_points[last_main_zz_idx + 3].pline.low < self.zz_points[last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if (
                    self.zz_points[last_main_zz_idx + 2].pline.low > self.zz_points[last_main_zz_idx].pline.low
                    and self.zz_points[last_main_zz_idx + 3].pline.high > self.zz_points[last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

    def update_main_zigzag(self):
        last_main_zz_idx = self.main_zz_idx[-1]
        while last_main_zz_idx + 3 < len(self.zz_points):
            last_main_zz_type = self.zz_points[last_main_zz_idx].ptype
            if last_main_zz_type == mta.POINT_TYPE.PEAK_POINT:
                if (
                    self.zz_points[last_main_zz_idx + 2].pline.high < self.zz_points[last_main_zz_idx].pline.high
                    and self.zz_points[last_main_zz_idx + 3].pline.low < self.zz_points[last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if (
                    self.zz_points[last_main_zz_idx + 2].pline.low > self.zz_points[last_main_zz_idx].pline.low
                    and self.zz_points[last_main_zz_idx + 3].pline.high
                    > self.zz_points[last_main_zz_idx + 1].pline.high
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

        last_main_zz_type = self.zz_points[last_main_zz_idx].ptype
        if last_main_zz_idx + 2 < len(self.zz_points):
            if last_main_zz_type == mta.POINT_TYPE.POKE_POINT:
                if self.zz_points[last_main_zz_idx + 2].pline.low < self.zz_points[last_main_zz_idx].pline.low:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if self.zz_points[last_main_zz_idx + 2].pline.high > self.zz_points[last_main_zz_idx].pline.high:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

        last_main_zz_type = self.zz_points[last_main_zz_idx].ptype
        if last_main_zz_idx + 1 < len(self.zz_points):
            last_kline = self.tfs_chart[self.tf].iloc[-1]
            if last_main_zz_type == mta.POINT_TYPE.POKE_POINT:
                if last_kline["Low"] < self.zz_points[last_main_zz_idx].pline.low:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if last_kline["High"] > self.zz_points[last_main_zz_idx].pline.high:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        chart = self.tfs_chart[self.tf]
        self.ma_vol.loc[len(self.ma_vol)] = ta.stream.SMA(chart["Volume"], self.params["ma_vol"])
        last_Zz_pidx = self.zz_points[-1].pidx
        mta.zigzag_stream(chart, self.min_zz_ratio, self.zz_points)
        last_main_idx = self.main_zz_idx[-1]
        self.update_main_zigzag()
        if last_main_idx != self.main_zz_idx[-1]:
            self.temp.append((self.zz_points[self.main_zz_idx[-1]].pidx, len(chart) - 1))

    def check_required_params(self):
        return all(
            [
                key in self.params.keys()
                for key in [
                    "ma_vol",
                    "vol_ratio_ma",
                    "kline_body_ratio",
                    "sl_fix_mode",
                ]
            ]
        )

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return True

    def update(self, tf):
        # update when new kline arrive
        super().update(tf)
        # check order signal
        self.check_close_signal()
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self):
        if len(self.main_zz_idx) < 5:
            return
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        p_1 = self.zz_points[self.main_zz_idx[-1]]
        p_2 = self.zz_points[self.main_zz_idx[-2]]
        p_3 = self.zz_points[self.main_zz_idx[-3]]
        p_4 = self.zz_points[self.main_zz_idx[-4]]
        p_5 = self.zz_points[self.main_zz_idx[-5]]
        if p_1.pidx in self.checked_pidx:
            return
        if p_1.ptype == mta.POINT_TYPE.POKE_POINT:
            #        /\
            #   /\  /  \/
            #  /  \/  2  1
            # / 4  3
            # 5
            if p_5.pline.low < p_3.pline.low <= p_1.pline.low and p_1.pline.low <= p_4.pline.high:
                if p_3.pline.low <= p_4.pline.high - 0.4 * (
                    p_4.pline.high - p_5.pline.low
                ) and p_3.pline.low >= p_4.pline.high - 0.62 * (p_4.pline.high - p_5.pline.low):
                    if 1.002 * p_4.pline.high < p_2.pline.high:
                        sl = p_1.pline.low
                        tp = None
                        order = Order(
                            OrderType.MARKET,
                            OrderSide.BUY,
                            last_kline["Close"],
                            tp=tp,
                            sl=sl,
                            status=OrderStatus.FILLED,
                        )
                        order["FILL_TIME"] = last_kline["Open time"]
                        order["strategy"] = self.name
                        order["description"] = self.description
                        order["desc"] = {}
                        order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                        if order:
                            self.trader.create_trade(order, self.volume)
                            self.orders_opening.append(order)
                            self.checked_pidx.append(p_1.pidx)
                            self.close_order_by_side(last_kline, OrderSide.SELL)
        else:
            if p_5.pline.high > p_3.pline.high >= p_1.pline.high and p_1.pline.high >= p_4.pline.low:
                if p_4.pline.low > p_2.pline.low * 1.002:
                    sl = p_1.pline.high
                    tp = None
                    order = Order(
                        OrderType.MARKET,
                        OrderSide.SELL,
                        last_kline["Close"],
                        tp=tp,
                        sl=sl,
                        status=OrderStatus.FILLED,
                    )
                    order["FILL_TIME"] = last_kline["Open time"]
                    order["strategy"] = self.name
                    order["description"] = self.description
                    order["desc"] = {}
                    order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                    if order:
                        self.trader.create_trade(order, self.volume)
                        self.orders_opening.append(order)
                        self.checked_pidx.append(p_1.pidx)
                        self.close_order_by_side(last_kline, OrderSide.BUY)

    def check_close_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        if last_kline["Volume"] < self.params["vol_ratio_ma"] * self.ma_vol.iloc[-1]:
            return
        if last_kline["Close"] < last_kline["Open"]:
            # red kline, check close buy orders
            if (last_kline["High"] - last_kline["Close"]) < 0.75 * (last_kline["High"] - last_kline["Low"]):
                return
        else:
            # green kline, check close sell orders
            if (last_kline["Close"] - last_kline["Low"]) < 0.75 * (last_kline["High"] - last_kline["Low"]):
                return

    def plot_orders(self):
        fig = make_subplots(2, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.8, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            yaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            xaxis=dict(showgrid=False),
            xaxis2=dict(showgrid=False),
            plot_bgcolor="rgb(19, 23, 34)",
            paper_bgcolor="rgb(121,125,127)",
            font=dict(color="rgb(247,249,249)"),
        )
        df = self.tfs_chart[self.tf]
        dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
        tmp_ot = df["Open time"]
        df["Open time"] = list(range(len(df)))
        super().plot_orders(fig, self.tf, 1, 1, dt2idx=dt2idx)

        fig.add_trace(
            go.Scatter(
                x=[df.iloc[ad.pidx]["Open time"] for ad in self.zz_points],
                y=[ad.pline.low if ad.ptype == mta.POINT_TYPE.POKE_POINT else ad.pline.high for ad in self.zz_points],
                mode="lines",
                line=dict(dash="dash"),
                marker_color=[
                    "rgba(255, 255, 0, 1)" if ad.ptype == mta.POINT_TYPE.POKE_POINT else "rgba(0, 255, 0, 1)"
                    for ad in self.zz_points
                ],
                name="ZigZag Point",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=ta.SMA(df["Close"], 200),
                mode="lines",
                line=dict(color="orange"),
                name="SMA_200",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        main_zz_points = [self.zz_points[i] for i in self.main_zz_idx]
        fig.add_trace(
            go.Scatter(
                x=[df.iloc[ad.pidx]["Open time"] for ad in main_zz_points],
                y=[ad.pline.low if ad.ptype == mta.POINT_TYPE.POKE_POINT else ad.pline.high for ad in main_zz_points],
                mode="lines",
                name="Main ZigZag Point",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        for x, y in self.temp:
            fig.add_shape(
                type="line",
                x0=df.iloc[x]["Open time"],
                y0=df.iloc[x]["High"],
                x1=df.iloc[y]["Open time"],
                y1=df.iloc[x]["High"],
                line=dict(color="red"),
                row=1,
                col=1,
            )

        for order in self.orders_closed:
            desc = order["desc"]

        colors = [
            "rgb(242, 54, 69)" if kline["Open"] > kline["Close"] else "rgb(8, 153, 129)" for i, kline in df.iterrows()
        ]
        fig.add_trace(
            go.Bar(x=list(range(len(df))), y=df["Volume"], marker_color=colors, marker_line_width=0, name="Volume"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.ma_vol,
                mode="lines",
                line=dict(color="rgba(0, 255, 0, 1)"),
                name="MA_VOL_{}".format(self.params["ma_vol"]),
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
        fig.update_layout(
            title={
                "text": "Breakout Strategy({}) (tf {})".format(self.trader.symbol_name, self.tf),
                "x": 0.5,
                "xanchor": "center",
            }
        )
        df["Open time"] = tmp_ot
        return fig
