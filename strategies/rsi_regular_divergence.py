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


class RSIRegularDivergence(BaseStrategy):
    """
    - RSI Divergence/Hidden Divergence strategy
    - Divergence: price move in opposite direction of technical indicator
        - Regular divergence
            - price creates higher highs but indicator creates lower highs
            - price creates lower low but indicator creates higher low
        -> trend change reversal signal
        - HIdden divergence
            - price creates higher low but indicator creates lower low
            - price creates lower high but dincator creates higer high
        -> trend continuation signals

    """

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.checked_pidx = []
        self.checked_seg = []
        self.divergenced_segs = []

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate ZigZag indicator
        chart = self.tfs_chart[self.tf]
        self.rsi = ta.RSI(chart["Close"], self.params["rsi_len"])
        self.delta_rsi = self.params["delta_rsi"]
        self.delta_price_ratio = 0.01 * self.params["delta_price_pct"]
        self.min_reward_ratio = 0.01 * self.params["min_rw_pct"]
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        if self.params["zz_type"] == "ZZ_CONV":
            self.zz_points = mta.zigzag_conv(chart, self.params["zz_conv_size"], self.min_zz_ratio)
        else:
            self.zz_points = mta.zigzag(chart, self.min_zz_ratio)
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        chart = self.tfs_chart[self.tf]
        self.rsi.loc[len(self.rsi)] = ta.stream.RSI(chart["Close"], self.params["rsi_len"])
        if self.params["zz_type"] == "ZZ_CONV":
            mta.zigzag_conv_stream(chart, self.params["zz_conv_size"], self.min_zz_ratio, self.zz_points)
        else:
            mta.zigzag_stream(chart, self.min_zz_ratio, self.zz_points)

    def check_required_params(self):
        return all(
            [
                key in self.params.keys()
                for key in [
                    "rsi_len",
                    "delta_rsi",
                    "delta_price_pct",
                    "n_last_point",
                    "min_rr",
                    "min_rw_pct",
                    "min_zz_pct",
                    "zz_type",
                    "zz_conv_size",
                    "sl_fix_mode",
                    "n_trend_point",
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
        # determind trend
        self.check_close_signal()
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def filter_poke_points(self, poke_points, last_point):
        valid_points = []
        for i, poke_point in enumerate(poke_points):
            slope, intercept = get_line_coffs(
                (poke_point.pidx, poke_point.pline.low), (last_point.pidx, last_point.pline.low)
            )
            b_1 = self.tfs_chart[self.tf].iloc[poke_point.pidx + 1 : last_point.pidx]["Low"]  # try replace with Low
            b_0 = self.tfs_chart[self.tf].iloc[poke_point.pidx + 1 : last_point.pidx].index.to_series()
            if ((slope * b_0 + intercept) * (1 - self.delta_price_ratio) <= b_1).all():
                valid_points.append(i)
        return valid_points

    def filter_peak_points(self, peak_points, last_point):
        valid_points = []
        for i, peak_point in enumerate(peak_points):
            slope, intercept = get_line_coffs(
                (peak_point.pidx, peak_point.pline.high), (last_point.pidx, last_point.pline.high)
            )
            b_1 = self.tfs_chart[self.tf].iloc[peak_point.pidx + 1 : last_point.pidx]["High"]  # try replace with Low
            b_0 = self.tfs_chart[self.tf].iloc[peak_point.pidx + 1 : last_point.pidx].index.to_series()
            if ((slope * b_0 + intercept) * (1 + self.delta_price_ratio) >= b_1).all():
                valid_points.append(i)
        return valid_points

    def check_rsi(self, rsi, above=True):
        idxs = rsi.index.to_series()
        slope, intercept = get_line_coffs((idxs.iloc[0], rsi.iloc[0]), (idxs.iloc[-1], rsi.iloc[-1]))
        b_1 = rsi.iloc[1:-1]
        b_0 = idxs.iloc[1:-1]
        if above:
            return (slope * b_0 + intercept - self.delta_rsi <= b_1).all()
        return (slope * b_0 + intercept + self.delta_rsi >= b_1).all()

    def find_divergenced_seg(self):
        last_zz_point = self.zz_points[-1]
        # Specify up/down trend lines using n_trend_point ZZPoint
        n_last_poke_points = []
        n_last_peak_points = []
        for zz_point in self.zz_points[-self.params["n_trend_point"] :]:
            if zz_point.ptype == mta.POINT_TYPE.POKE_POINT:
                n_last_poke_points.append((zz_point.pidx, zz_point.pline.low))
            else:
                n_last_peak_points.append((zz_point.pidx, zz_point.pline.high))
        if len(n_last_peak_points) < 2 or len(n_last_poke_points) < 2:
            return
        if last_zz_point.ptype == mta.POINT_TYPE.PEAK_POINT:
            # price higher highs, indicator lower highs.
            # get n_last_point peak points lower high
            n_zz_points = []
            i = 1
            while 2 * i + 1 <= len(self.zz_points):
                current_zz_point = self.zz_points[-(2 * i + 1)]
                if current_zz_point.pline.high < last_zz_point.pline.high:
                    n_zz_points.append(current_zz_point)
                if len(n_zz_points) >= self.params["n_last_point"]:
                    break
                i += 1
            valid_points_idx = self.filter_peak_points(n_zz_points, last_zz_point)
            for valid_point_idx in valid_points_idx:
                zz_point = n_zz_points[valid_point_idx]
                low_2_idx = self.rsi.iloc[last_zz_point.pidx :].idxmax()
                if self.rsi[zz_point.pidx] > self.rsi[low_2_idx] + self.delta_rsi:
                    if (last_zz_point.pline.high - zz_point.pline.high) < self.delta_price_ratio * zz_point.pline.high:
                        continue
                    if not self.check_rsi(self.rsi.iloc[zz_point.pidx : low_2_idx + 1], above=False):
                        continue
                    self.divergenced_segs.append(
                        ((zz_point, last_zz_point), (self.rsi[zz_point.pidx], self.rsi[low_2_idx]))
                    )
                    break

        if last_zz_point.ptype == mta.POINT_TYPE.POKE_POINT:
            # price lower low, indicator higher low.
            # get n_last_point peak points lower high
            n_zz_points = []
            i = 1
            while 2 * i + 1 <= len(self.zz_points):
                current_zz_point = self.zz_points[-(2 * i + 1)]
                if current_zz_point.pline.low > last_zz_point.pline.low:
                    n_zz_points.append(current_zz_point)
                if len(n_zz_points) >= self.params["n_last_point"]:
                    break
                i += 1
            valid_points_idx = self.filter_poke_points(n_zz_points, last_zz_point)
            for valid_point_idx in valid_points_idx:
                zz_point = n_zz_points[valid_point_idx]
                low_2_idx = self.rsi.iloc[last_zz_point.pidx :].idxmin()
                if self.rsi[zz_point.pidx] < self.rsi[low_2_idx] - self.delta_rsi:
                    if (zz_point.pline.low - last_zz_point.pline.low) < self.delta_price_ratio * zz_point.pline.low:
                        continue
                    if not self.check_rsi(self.rsi.iloc[zz_point.pidx : low_2_idx + 1], above=True):
                        continue
                    self.divergenced_segs.append(
                        ((zz_point, last_zz_point), (self.rsi[zz_point.pidx], self.rsi[low_2_idx]))
                    )
                    break

    def check_signal(self):
        last_zz_point = self.zz_points[-1]
        if last_zz_point.pidx in self.checked_pidx:
            return
        self.checked_pidx.append(last_zz_point.pidx)
        self.find_divergenced_seg()
        if len(self.divergenced_segs) == 0:
            return
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        last_diverg_seg, last_rsi_diverg = self.divergenced_segs[-1]
        if last_diverg_seg[1].pidx in self.checked_seg:
            return
        # Specify up/down trend lines using n_trend_point ZZPoint
        n_last_poke_points = []
        n_last_peak_points = []
        for zz_point in self.zz_points[-self.params["n_trend_point"] :]:
            if zz_point.ptype == mta.POINT_TYPE.POKE_POINT:
                n_last_poke_points.append((zz_point.pidx, zz_point.pline.low))
            else:
                n_last_peak_points.append((zz_point.pidx, zz_point.pline.high))
        if len(n_last_peak_points) < 2 or len(n_last_poke_points) < 2:
            return
        max_high = max([zp[1] for zp in n_last_peak_points])
        min_low = min([zp[1] for zp in n_last_poke_points])
        if last_zz_point.ptype == mta.POINT_TYPE.PEAK_POINT and last_diverg_seg[0].ptype == mta.POINT_TYPE.PEAK_POINT:
            if last_zz_point.pline.high > self.zz_points[-3].pline.high:
                return
            # price higher highs, indicator lower highs.
            sl = last_zz_point.pline.high
            entry = last_kline["Close"]
            tp = max_high - self.params["fib_retr_lv"] * (max_high - min_low)
            order = Order(
                OrderType.MARKET,
                OrderSide.SELL,
                entry,
                tp=tp,
                sl=sl,
                status=OrderStatus.FILLED,
            )
            if order.rr >= self.params["min_rr"] and order.reward_ratio > self.min_reward_ratio and order.is_valid():
                if order.type == OrderType.MARKET:
                    order["FILL_TIME"] = last_kline["Open time"]
                order["strategy"] = self.name
                order["description"] = self.description
                self.trader.create_trade(order, self.volume)
                order["desc"] = {
                    "type": "higher_high_lower_high",
                    "zz_point_1": last_diverg_seg[0],
                    "zz_point_2": last_diverg_seg[1],
                    "rsi_1": last_rsi_diverg[0],
                    "rsi_2": last_rsi_diverg[1],
                }
                self.orders_opening.append(order)
                self.checked_seg.append(last_diverg_seg[1].pidx)

        elif last_zz_point.ptype == mta.POINT_TYPE.POKE_POINT and last_diverg_seg[0].ptype == mta.POINT_TYPE.POKE_POINT:
            if last_zz_point.pline.low < self.zz_points[-3].pline.low:
                return
            # price lower low, indicator higher low.
            sl = last_zz_point.pline.low
            entry = last_kline["Close"]
            tp = min_low + self.params["fib_retr_lv"] * (max_high - min_low)
            order = Order(
                OrderType.MARKET,
                OrderSide.BUY,
                entry,
                tp=tp,
                sl=sl,
                status=OrderStatus.FILLED,
            )
            if order.rr >= self.params["min_rr"] and order.reward_ratio > self.min_reward_ratio and order.is_valid():
                if order.type == OrderType.MARKET:
                    order["FILL_TIME"] = last_kline["Open time"]
                order["strategy"] = self.name
                order["description"] = self.description
                self.trader.create_trade(order, self.volume)
                order["desc"] = {
                    "type": "lower_low_higher_low",
                    "zz_point_1": last_diverg_seg[0],
                    "zz_point_2": last_diverg_seg[1],
                    "rsi_1": last_rsi_diverg[0],
                    "rsi_2": last_rsi_diverg[1],
                }
                self.orders_opening.append(order)
                self.checked_seg.append(last_diverg_seg[1].pidx)

    def check_close_signal(self):
        if len(self.zz_points) < 4:
            return
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        zz_point_1 = self.zz_points[-1]
        if self.zz_points[-1].ptype == mta.POINT_TYPE.PEAK_POINT:
            if self.zz_points[-1].pline.high < self.zz_points[-3].pline.high:
                if self.zz_points[-2].pline.low < self.zz_points[-4].pline.low:
                    self.close_order_by_side(last_kline, OrderSide.BUY)
            else:
                if self.zz_points[-2].pline.low > self.zz_points[-4].pline.low:
                    self.close_order_by_side(last_kline, OrderSide.SELL)
        else:
            if self.zz_points[-1].pline.low > self.zz_points[-3].pline.low:
                if self.zz_points[-2].pline.high > self.zz_points[-4].pline.high:
                    self.close_order_by_side(last_kline, OrderSide.SELL)
            else:
                if self.zz_points[-2].pline.high < self.zz_points[-3].pline.high:
                    self.close_order_by_side(last_kline, OrderSide.BUY)

    def plot_orders(self):
        fig = make_subplots(3, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis=dict(showgrid=False),
            xaxis2=dict(showgrid=False),
            yaxis=dict(showgrid=False),
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
                x=[df.iloc[ad.pidx]["Open time"] for ad in self.zz_points],
                y=[ad.pline.low if ad.ptype == mta.POINT_TYPE.POKE_POINT else ad.pline.high for ad in self.zz_points],
                mode="lines",
                line=dict(color="orange", dash="dash"),
                hoverinfo="skip",
                name="ZigZag Point",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.rsi,
                mode="lines",
                line=dict(color="orange"),
                name="RSI_{}".format(self.params["rsi_len"]),
            ),
            row=2,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=30,
            x1=1,
            y1=30,
            xref="x domain",
            line=dict(color="magenta", width=1, dash="dash"),
            row=2,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=70,
            x1=1,
            y1=70,
            xref="x domain",
            line=dict(color="magenta", width=1, dash="dash"),
            row=2,
            col=1,
        )
        for order in self.orders_closed:
            desc = order["desc"]
            zz_point_1 = desc["zz_point_1"]
            zz_point_2 = desc["zz_point_2"]
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_point_1.pidx]["Open time"], df.iloc[zz_point_2.pidx]["Open time"]],
                    y=[desc["rsi_1"], desc["rsi_2"]],
                    mode="markers+lines",
                    line=dict(color="orange", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_point_1.pidx]["Open time"], df.iloc[zz_point_2.pidx]["Open time"]],
                    y=[zz_point_1.pline.low, zz_point_2.pline.low]
                    if zz_point_1.ptype == mta.POINT_TYPE.POKE_POINT
                    else [zz_point_1.pline.high, zz_point_2.pline.high],
                    mode="markers+lines",
                    line=dict(color="orange", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        colors = ["red" if kline["Open"] > kline["Close"] else "green" for i, kline in df.iterrows()]
        fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], marker_color=colors), row=3, col=1)

        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))

        fig.update_layout(
            title={
                "text": "RSI Regular Divergence Strategy (symbol: {}, tf: {})".format(self.trader.symbol_name, self.tf),
                "x": 0.5,
                "xanchor": "center",
            }
        )
        df["Open time"] = tmp_ot
        return fig
