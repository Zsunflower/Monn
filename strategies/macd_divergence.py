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


class MACDDivergence(BaseStrategy):
    """
    - MACD Divergence/Hidden Divergence strategy
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

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate ZigZag indicator
        chart = self.tfs_chart[self.tf]
        self.macd, self.macdsignal, self.macdhist = ta.MACD(
            chart["Close"],
            self.params["macd_inputs"]["fast_len"],
            self.params["macd_inputs"]["slow_len"],
            self.params["macd_inputs"]["signal"],
        )
        if self.params["macd_type"] == "MACD":
            self.macd_type = self.macd
        elif self.params["macd_type"] == "MACD_SIGNAL":
            self.macd_type = self.macdsignal
        else:
            self.macd_type = self.macdhist

        self.delta_macd = self.params["delta_macd"]
        self.delta_price_ratio = 0.01 * self.params["delta_price_pct"]
        self.min_reward_ratio = 0.01 * self.params["min_rw_pct"]
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        self.min_trend_ratio = 0.01 * self.params["min_trend_pct"]
        if self.params["zz_type"] == "ZZ_CONV":
            self.zz_points = mta.zigzag_conv(chart, self.params["zz_conv_size"], self.min_zz_ratio)
        else:
            self.zz_points = mta.zigzag(chart, self.min_zz_ratio)
        self.last_check_zz_point = self.zz_points[0]
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        chart = self.tfs_chart[self.tf]
        macd, macdsignal, macdhist = ta.stream.MACD(
            chart["Close"],
            self.params["macd_inputs"]["fast_len"],
            self.params["macd_inputs"]["slow_len"],
            self.params["macd_inputs"]["signal"],
        )
        self.macd.loc[len(self.macd)] = macd
        self.macdsignal.loc[len(self.macdsignal)] = macdsignal
        self.macdhist.loc[len(self.macdhist)] = macdhist

        if self.params["zz_type"] == "ZZ_CONV":
            mta.zigzag_conv_stream(chart, self.params["zz_conv_size"], self.min_zz_ratio, self.zz_points)
        else:
            mta.zigzag_stream(chart, self.min_zz_ratio, self.zz_points)

    def check_required_params(self):
        return all(
            [
                key in self.params.keys()
                for key in [
                    "macd_inputs",
                    "delta_macd",
                    "delta_price_pct",
                    "n_last_point",
                    "min_rr",
                    "min_rw_pct",
                    "min_zz_pct",
                    "min_trend_pct",
                    "min_updown_ratio",
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

    def check_macd(self, macd, above=True):
        idxs = macd.index.to_series()
        slope, intercept = get_line_coffs((idxs.iloc[0], macd.iloc[0]), (idxs.iloc[-1], macd.iloc[-1]))
        b_1 = macd.iloc[1:-1]
        b_0 = idxs.iloc[1:-1]
        if above:
            return (slope * b_0 + intercept - self.delta_macd <= b_1).all()
        return (slope * b_0 + intercept + self.delta_macd >= b_1).all()

    def check_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        last_zz_point = self.zz_points[-1]
        if last_zz_point.pidx in self.checked_pidx:
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
        self.up_trend_line = find_uptrend_line(n_last_poke_points)
        self.down_trend_line = find_downtrend_line(n_last_peak_points)
        min_pidx = min(n_last_peak_points[0][0], n_last_poke_points[0][0])

        up_pct = (self.up_trend_line[1][1] - self.up_trend_line[0][1]) / self.up_trend_line[0][1]
        down_pct = (self.down_trend_line[1][1] - self.down_trend_line[0][1]) / self.down_trend_line[0][1]
        y_up = get_y_on_line(self.up_trend_line, len(chart) - 1)
        y_down = get_y_on_line(self.down_trend_line, len(chart) - 1)
        up_close_ratio = abs((last_kline["Close"] - y_up) / (y_down - y_up))
        down_close_ratio = abs((last_kline["Close"] - y_down) / (y_down - y_up))

        def get_next_zz_point(zz_point):
            for j in range(1, len(self.zz_points)):
                if self.zz_points[-j].pidx == zz_point.pidx:
                    return self.zz_points[-j + 1].pidx
            return self.zz_points[-1].pidx

        if last_zz_point.ptype == mta.POINT_TYPE.POKE_POINT:
            # check for higher low, lower low
            # if chart.iloc[last_zz_point.pidx + 1 :]["Close"].min() < last_zz_point.pline.high:
            #     self.checked_pidx.append(last_zz_point.pidx)
            #     return
            # check trend
            if down_pct < 0:
                self.checked_pidx.append(last_zz_point.pidx)
                return
            # check updown_ratio
            if up_close_ratio > self.params["min_updown_ratio"]:
                return
            # check macd, macdsignal > 0
            if self.macd.iloc[-1] < self.macdsignal.iloc[-1]:
                return
            # get n_last_point poke points lower low
            n_zz_points = []
            i = 1
            while 2 * i + 1 <= len(self.zz_points):
                current_zz_point = self.zz_points[-(2 * i + 1)]
                if current_zz_point.pline.low < last_zz_point.pline.low:
                    n_zz_points.append(current_zz_point)
                if len(n_zz_points) >= self.params["n_last_point"]:
                    break
                i += 1

            valid_points_idx = self.filter_poke_points(n_zz_points, last_zz_point)
            for valid_point_idx in valid_points_idx:
                zz_point = n_zz_points[valid_point_idx]
                if (last_zz_point.pline.low - zz_point.pline.low) < self.delta_price_ratio * zz_point.pline.low:
                    continue
                if zz_point.pidx < min_pidx:
                    continue
                max_high_idx = get_next_zz_point(zz_point)
                low_1_idx = self.macd_type.iloc[zz_point.pidx : max_high_idx].idxmin()
                low_2_idx = self.macd_type.iloc[self.zz_points[-2].pidx : len(chart)].idxmin()
                if self.macd_type.iloc[low_1_idx] > self.macd_type.iloc[low_2_idx] + self.delta_macd:
                    if self.macd_type.iloc[low_1_idx] * self.macd_type.iloc[low_2_idx] < 0:
                        continue
                    if not self.check_macd(self.macd_type.iloc[low_1_idx : low_2_idx + 1], above=True):
                        continue
                    sl = min(chart.iloc[last_zz_point.pidx :]["Low"].min(), last_zz_point.pline.low)
                    sl = (1 - self.delta_price_ratio) * sl
                    if (
                        self.down_trend_line[0][1] - self.up_trend_line[0][1]
                        < self.down_trend_line[1][1] - self.up_trend_line[1][1]
                    ):
                        tp = get_y_on_line(self.down_trend_line, len(chart))
                    else:
                        tp = get_y_on_line(self.down_trend_line, len(chart) + 10)
                    order = Order(
                        OrderType.MARKET,
                        OrderSide.BUY,
                        last_kline["Close"],
                        tp=tp,
                        sl=sl,
                        status=OrderStatus.FILLED,
                    )
                    order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                    if order is None:
                        continue
                    if (
                        order.rr >= self.params["min_rr"]
                        and order.reward_ratio > self.min_reward_ratio
                        and order.is_valid()
                    ):
                        if order.type == OrderType.MARKET:
                            order["FILL_TIME"] = last_kline["Open time"]
                        order["strategy"] = self.name
                        order["description"] = self.description
                        self.trader.create_trade(order, self.volume)
                        order["desc"] = {
                            "type": "higher_low_lower_low",
                            "zz_point_1": zz_point,
                            "zz_point_2": last_zz_point,
                            "up_trend_line": self.up_trend_line,
                            "down_trend_line": self.down_trend_line,
                            "up_pct": up_pct,
                            "down_pct": down_pct,
                            "updown_close_ratio": up_close_ratio,
                            "idx_1": low_1_idx,
                            "idx_2": low_2_idx,
                        }
                        self.orders_opening.append(order)
                        self.checked_pidx.append(last_zz_point.pidx)
                        break

        if last_zz_point.ptype == mta.POINT_TYPE.PEAK_POINT:
            # check for lower high, higher high
            # if chart.iloc[last_zz_point.pidx + 1 :]["Close"].max() > last_zz_point.pline.low:
            #     self.checked_pidx.append(last_zz_point.pidx)
            #     return
            # check trend
            if up_pct > 0:
                self.checked_pidx.append(last_zz_point.pidx)
                return
            # check updown_ratio
            if down_close_ratio > self.params["min_updown_ratio"]:
                return
            # check macd, macdsignal < 0
            if self.macd.iloc[-1] > self.macdsignal.iloc[-1]:
                return
            # get n_last_point peak points higher high
            n_zz_points = []
            i = 1
            while 2 * i + 1 <= len(self.zz_points):
                current_zz_point = self.zz_points[-(2 * i + 1)]
                if current_zz_point.pline.high > last_zz_point.pline.high:
                    n_zz_points.append(current_zz_point)
                if len(n_zz_points) >= self.params["n_last_point"]:
                    break
                i += 1

            valid_points_idx = self.filter_peak_points(n_zz_points, last_zz_point)
            for valid_point_idx in valid_points_idx:
                zz_point = n_zz_points[valid_point_idx]
                if (zz_point.pline.high - last_zz_point.pline.high) < self.delta_price_ratio * zz_point.pline.high:
                    continue
                if zz_point.pidx < min_pidx:
                    continue
                min_low_idx = get_next_zz_point(zz_point)
                low_1_idx = self.macd_type.iloc[zz_point.pidx : min_low_idx].idxmax()
                low_2_idx = self.macd_type.iloc[self.zz_points[-2].pidx : len(chart)].idxmax()
                if self.macd_type.iloc[low_1_idx] < self.macd_type.iloc[low_2_idx] - self.delta_macd:
                    if self.macd_type.iloc[low_1_idx] * self.macd_type.iloc[low_2_idx] < 0:
                        continue
                    if not self.check_macd(self.macd_type.iloc[low_1_idx : low_2_idx + 1], above=False):
                        continue
                    sl = max(chart.iloc[last_zz_point.pidx :]["High"].max(), last_zz_point.pline.high)
                    sl = (1 + self.delta_price_ratio) * sl
                    if (
                        self.down_trend_line[0][1] - self.up_trend_line[0][1]
                        < self.down_trend_line[1][1] - self.up_trend_line[1][1]
                    ):
                        tp = get_y_on_line(self.up_trend_line, len(chart))
                    else:
                        tp = get_y_on_line(self.up_trend_line, len(chart) + 10)
                    order = Order(
                        OrderType.MARKET,
                        OrderSide.SELL,
                        last_kline["Close"],
                        tp=tp,
                        sl=sl,
                        status=OrderStatus.FILLED,
                    )
                    order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                    if order is None:
                        continue
                    if (
                        order.rr >= self.params["min_rr"]
                        and order.reward_ratio > self.min_reward_ratio
                        and order.is_valid()
                    ):
                        if order.type == OrderType.MARKET:
                            order["FILL_TIME"] = last_kline["Open time"]
                        order["strategy"] = self.name
                        order["description"] = self.description
                        self.trader.create_trade(order, self.volume)
                        order["desc"] = {
                            "type": "lower_high_higher_high",
                            "zz_point_1": zz_point,
                            "zz_point_2": last_zz_point,
                            "up_trend_line": self.up_trend_line,
                            "down_trend_line": self.down_trend_line,
                            "up_pct": up_pct,
                            "down_pct": down_pct,
                            "updown_close_ratio": down_close_ratio,
                            "idx_1": low_1_idx,
                            "idx_2": low_2_idx,
                        }
                        self.orders_opening.append(order)
                        self.checked_pidx.append(last_zz_point.pidx)
                        break

    def plot_orders(self):
        fig = make_subplots(2, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.6, 0.4])
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
                line=dict(color="orange", width=1, dash="dash"),
                name="ZigZag Point",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"], y=self.macd, mode="lines", line=dict(color="rgba(0, 255, 0, 1)"), name="MACD"
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.macdsignal,
                mode="lines",
                line=dict(color="rgba(255, 0, 0, 1)"),
                name="MACD Signal",
            ),
            row=2,
            col=1,
        )
        colors = ["red" if mh < 0 else "green" for mh in self.macdhist]
        fig.add_trace(
            go.Bar(x=df["Open time"], y=self.macdhist, marker_color=colors, name="MACD Histogram"), row=2, col=1
        )

        for order in self.orders_closed:
            desc = order["desc"]
            zz_point_1 = desc["zz_point_1"]
            zz_point_2 = desc["zz_point_2"]
            up_trend = desc["up_trend_line"]
            down_trend = desc["down_trend_line"]
            fig.add_shape(
                type="line",
                x0=df.iloc[up_trend[0][0]]["Open time"],
                y0=up_trend[0][1],
                x1=df.iloc[up_trend[1][0]]["Open time"],
                y1=up_trend[1][1],
                line=dict(color="green", width=1),
                row=1,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=df.iloc[down_trend[0][0]]["Open time"],
                y0=down_trend[0][1],
                x1=df.iloc[down_trend[1][0]]["Open time"],
                y1=down_trend[1][1],
                line=dict(color="red", width=1),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.iloc[up_trend[0][0]]["Open time"],
                        df.iloc[up_trend[1][0]]["Open time"],
                        df.iloc[down_trend[1][0]]["Open time"],
                        df.iloc[down_trend[0][0]]["Open time"],
                        df.iloc[up_trend[0][0]]["Open time"],
                    ],
                    y=[up_trend[0][1], up_trend[1][1], down_trend[1][1], down_trend[0][1], up_trend[0][1]],
                    mode="lines",
                    line=dict(color="rgba(156, 39, 176, 0.8)", width=1),
                    fill="toself",
                    fillcolor="rgba(128, 0, 128, 0.2)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_point_1.pidx]["Open time"], df.iloc[zz_point_2.pidx]["Open time"]],
                    y=[zz_point_1.pline.low, zz_point_2.pline.low]
                    if zz_point_1.ptype == mta.POINT_TYPE.POKE_POINT
                    else [zz_point_1.pline.high, zz_point_2.pline.high],
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[desc["idx_1"]]["Open time"], df.iloc[desc["idx_2"]]["Open time"]],
                    y=[self.macd_type.iloc[desc["idx_1"]], self.macd_type.iloc[desc["idx_2"]]],
                    mode="markers+lines",
                    line=dict(color="red", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                    marker_symbol="x-thin-open",
                    marker_color="white",
                    marker_line_width=2,
                    marker_size=10,
                ),
                row=2,
                col=1,
            )

        # colors = ["red" if kline["Open"] > kline["Close"] else "green" for i, kline in df.iterrows()]
        # fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], marker_color=colors), row=3, col=1)

        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))

        fig.update_layout(
            title={
                "text": "MACD Divergence/Hidden Divergence Strategy (symbol: {}, tf: {})".format(
                    self.trader.symbol_name, self.tf
                ),
                "x": 0.5,
                "xanchor": "center",
            }
        )
        df["Open time"] = tmp_ot
        return fig
