import logging
from enum import Enum
import pandas as pd
from datetime import datetime
from typing import List
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus
from utils import find_uptrend_line, find_downtrend_line, get_y_on_line


bot_logger = logging.getLogger("bot_logger")


class TLStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    STOPPED = "STOPPED"


class TLType(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class TrendLine:
    __trend_id__ = 1

    def __init__(self, zz_points):
        self.trend_id = TrendLine.__trend_id__
        TrendLine.__trend_id__ += 1
        self.zz_points = zz_points
        self.find_bound()

    def find_bound(self):
        # find upper/lower bound of the trend line
        if self.zz_points[-1].ptype == mta.POINT_TYPE.PEAK_POINT:
            self.type = TLType.RESISTANCE
            self.upper_line = find_downtrend_line([(zz_point.pidx, zz_point.pline.high) for zz_point in self.zz_points])
            self.upper_line = [list(self.upper_line[0]), list(self.upper_line[1])]
            self.lower_line = [list(self.upper_line[0]), list(self.upper_line[1])]
            delta_lower = min(
                [get_y_on_line(self.upper_line, zz_point.pidx) - zz_point.pline.low for zz_point in self.zz_points]
            )
            self.lower_line[0][1] -= delta_lower
            self.lower_line[1][1] -= delta_lower
        else:
            self.type = TLType.SUPPORT
            self.lower_line = find_uptrend_line([(zz_point.pidx, zz_point.pline.low) for zz_point in self.zz_points])
            self.lower_line = [list(self.lower_line[0]), list(self.lower_line[1])]
            self.upper_line = [list(self.lower_line[0]), list(self.lower_line[1])]
            delta_upper = min(
                [zz_point.pline.high - get_y_on_line(self.lower_line, zz_point.pidx) for zz_point in self.zz_points]
            )
            self.upper_line[0][1] += delta_upper
            self.upper_line[1][1] += delta_upper
        self.status = TLStatus.ACTIVE

    def append(self, zz_point):
        # append a zz_point to the trend line
        self.zz_points.append(zz_point)
        self.find_bound()

    def is_pass_zz_point(self, zz_point, delta):
        y_lower = get_y_on_line(self.lower_line, zz_point.pidx)
        y_upper = get_y_on_line(self.upper_line, zz_point.pidx)
        if zz_point.ptype == mta.POINT_TYPE.PEAK_POINT:
            return zz_point.pline.high + delta >= y_lower and zz_point.pline.low - delta <= y_upper
        return zz_point.pline.low - delta <= y_upper and zz_point.pline.high + delta >= y_lower

    def get_passed_zz_idx(self, zz_points, delta):
        idx = []
        for i, zz_point in enumerate(zz_points):
            if self.is_pass_zz_point(zz_point, delta):
                idx.append(i)
        return idx

    def keep_zz_idx(self, idx):
        self.zz_points = [self.zz_points[i] for i in idx]

    def get_y_on_lower_line(self, pidx):
        return get_y_on_line(self.lower_line, pidx)

    def get_y_on_upper_line(self, pidx):
        return get_y_on_line(self.upper_line, pidx)

    def clone(self):
        return TrendLine(self.zz_points[:])

    def is_above_zz_point(self, zz_point):
        return get_y_on_line(self.lower_line, zz_point.pidx) >= zz_point.pline.high

    def is_below_zz_point(self, zz_point):
        return get_y_on_line(self.upper_line, zz_point.pidx) <= zz_point.pline.low


class TrendFollowing(BaseStrategy):
    """
    - Trend Following strategy
    """

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        self.min_order_zz_ratio = 0.01 * self.params["min_order_zz_pct"]
        self.max_last_trend_line = self.params["max_last_trend_line"]
        self.max_last_zigzag = self.params["max_last_zigzag"]
        self.temp = []
        self.checked_pidx = []
        self.trend_lines: List[TrendLine] = []

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate HA candelstick
        chart = self.tfs_chart[self.tf]
        self.zz_points = mta.zigzag(chart, self.min_zz_ratio)
        self.order_zz_points = mta.zigzag(chart, self.min_order_zz_ratio)
        self.init_main_zigzag()
        self.find_trend()
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
        last_Zz_pidx = self.zz_points[-1].pidx
        mta.zigzag_stream(chart, self.min_zz_ratio, self.zz_points)
        mta.zigzag_stream(chart, self.min_order_zz_ratio, self.order_zz_points)
        last_main_idx = self.main_zz_idx[-1]
        self.update_main_zigzag()
        if last_main_idx != self.main_zz_idx[-1]:
            self.temp.append((self.zz_points[self.main_zz_idx[-1]].pidx, len(chart) - 1))
            self.adjust_sl()
            self.find_trend()

    def check_required_params(self):
        return all(
            [
                key in self.params.keys()
                for key in [
                    "min_zz_pct",
                    "min_order_zz_pct",
                    "delta_zz",
                    "delta_order",
                    "max_last_trend_line",
                    "sl_fix_mode",
                ]
            ]
        )

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return True

    def find_trend(self):
        last_main_zz = self.zz_points[self.main_zz_idx[-1]]
        new_trend_lines = []
        for i, trend_line in enumerate(reversed(self.trend_lines)):
            if trend_line.is_pass_zz_point(last_main_zz, self.params["delta_zz"]):
                trend_line.append(last_main_zz)
            if i >= 2 * self.max_last_trend_line - 1:
                break

        zz_points = []
        i = 1
        while i <= len(self.main_zz_idx):
            if self.zz_points[self.main_zz_idx[-i]].ptype == last_main_zz.ptype:
                zz_points = [self.zz_points[self.main_zz_idx[-i]]] + zz_points
                if len(zz_points) >= 3:
                    trend_line = TrendLine(zz_points)
                    passed_pidxs = trend_line.get_passed_zz_idx(zz_points, self.params["delta_zz"])
                    if len(passed_pidxs) >= 3 and trend_line.is_pass_zz_point(zz_points[-1], self.params["delta_zz"]):
                        trend_line.keep_zz_idx(passed_pidxs)
                        self.trend_lines.append(trend_line)
                        break
                if len(zz_points) >= self.max_last_zigzag:
                    break
            i += 1

    def update(self, tf):
        # update when new kline arrive
        super().update(tf)
        # check order signal
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        self.adjust_sl_to_entry(last_kline)
        if len(self.main_zz_idx) < 4:
            return
        last_zz_point = self.order_zz_points[-1]
        if last_zz_point.pidx in self.checked_pidx:
            return
        self.checked_pidx.append(last_zz_point.pidx)

        for i, trend_line in enumerate(reversed(self.trend_lines)):
            if trend_line.is_pass_zz_point(last_zz_point, self.params["delta_order"]):
                if last_zz_point.ptype == mta.POINT_TYPE.POKE_POINT:
                    self.close_order_by_side(last_kline, OrderSide.SELL)
                    sl = min(last_zz_point.pline.low, trend_line.get_y_on_lower_line(last_zz_point.pidx))
                    tp = None
                    order = Order(
                        OrderType.MARKET,
                        OrderSide.BUY,
                        last_kline["Close"],
                        tp=tp,
                        sl=sl,
                        status=OrderStatus.FILLED,
                    )
                    stop_price = (
                        self.zz_points[self.main_zz_idx[-1]].pline.high
                        if self.zz_points[self.main_zz_idx[-1]].ptype == mta.POINT_TYPE.PEAK_POINT
                        else self.zz_points[self.main_zz_idx[-2]].pline.high
                    )
                    stop_price = max(stop_price, order.entry + 3 * (order.entry - order.sl))
                else:
                    self.close_order_by_side(last_kline, OrderSide.BUY)
                    sl = max(last_zz_point.pline.high, trend_line.get_y_on_upper_line(last_zz_point.pidx))
                    tp = None
                    order = Order(
                        OrderType.MARKET,
                        OrderSide.SELL,
                        last_kline["Close"],
                        tp=tp,
                        sl=sl,
                        status=OrderStatus.FILLED,
                    )
                    stop_price = (
                        self.zz_points[self.main_zz_idx[-1]].pline.low
                        if self.zz_points[self.main_zz_idx[-1]].ptype == mta.POINT_TYPE.POKE_POINT
                        else self.zz_points[self.main_zz_idx[-2]].pline.low
                    )
                    stop_price = min(stop_price, order.entry + 3 * (order.entry - order.sl))
                order["strategy"] = self.name
                order["description"] = self.description
                order["desc"] = {
                    "trend_line": trend_line.clone(),
                    "trend_line_id": trend_line.trend_id,
                    "pidx": last_zz_point.pidx,
                    "stop_price": stop_price,
                }
                order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
                if order:
                    if order.type == OrderType.MARKET:
                        order["FILL_TIME"] = last_kline["Open time"]
                    else:
                        order["LIMIT_TIME"] = last_kline["Open time"]
                    self.trader.create_trade(order, self.volume)
                    self.orders_opening.append(order)
                    break
            if i >= 2 * self.max_last_trend_line - 1:
                break

    def adjust_sl(self):
        last_main_zz = self.zz_points[self.main_zz_idx[-1]]
        if last_main_zz.ptype == mta.POINT_TYPE.PEAK_POINT:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.status != OrderStatus.FILLED:
                    continue
                if order.side == OrderSide.BUY:
                    continue
                if not order.has_sl() or order.sl > last_main_zz.pline.high:
                    order.adjust_sl(last_main_zz.pline.high)
                    self.trader.adjust_sl(order, last_main_zz.pline.high)
        else:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.status != OrderStatus.FILLED:
                    continue
                if order.side == OrderSide.SELL:
                    continue
                if not order.has_sl() or order.sl < last_main_zz.pline.low:
                    order.adjust_sl(last_main_zz.pline.low)
                    self.trader.adjust_sl(order, last_main_zz.pline.low)

    def adjust_sl_to_entry(self, last_kline):
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            if order.status == OrderStatus.FILLED:
                if order.side == OrderSide.BUY:
                    new_sl = order.entry + 0.45 * (order["desc"]["stop_price"] - order.entry)
                    if not order.has_sl() or order.sl < new_sl:
                        if (last_kline["Close"] - order.entry) > 0.9 * (order["desc"]["stop_price"] - order.entry):
                            order.adjust_sl(new_sl)
                            self.trader.adjust_sl(order, new_sl)
                else:
                    new_sl = order.entry - 0.45 * (order.entry - order["desc"]["stop_price"])
                    if not order.has_sl() or order.sl > new_sl:
                        if (order.entry - last_kline["Close"]) > 0.9 * (order.entry - order["desc"]["stop_price"]):
                            order.adjust_sl(new_sl)
                            self.trader.adjust_sl(order, new_sl)

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
                x=[df.iloc[ad.pidx]["Open time"] for ad in self.order_zz_points],
                y=[
                    ad.pline.low if ad.ptype == mta.POINT_TYPE.POKE_POINT else ad.pline.high
                    for ad in self.order_zz_points
                ],
                mode="lines",
                line=dict(color="rgba(255, 255, 255, 1)", width=1),
                name="ZigZag Point",
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

        for i, trend_line in enumerate(self.trend_lines):
            lower_line = trend_line.lower_line
            upper_line = trend_line.upper_line
            zz_points = trend_line.zz_points
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.iloc[lower_line[0][0]]["Open time"],
                        df.iloc[lower_line[1][0]]["Open time"],
                        df.iloc[upper_line[1][0]]["Open time"],
                        df.iloc[upper_line[0][0]]["Open time"],
                        df.iloc[lower_line[0][0]]["Open time"],
                    ],
                    y=[
                        lower_line[0][1],
                        lower_line[1][1],
                        upper_line[1][1],
                        upper_line[0][1],
                        lower_line[0][1],
                    ],
                    mode="lines",
                    line=dict(color="rgba(0, 255, 0, 1)", width=1),
                    fill="toself",
                    fillcolor="rgba(128, 0, 128, 0.25)",
                    hoverinfo="skip",
                    showlegend=True,
                    name="trend_line_{}".format(i),
                    legendgroup="trend_line",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_point.pidx]["Open time"] for zz_point in zz_points],
                    y=[
                        0.5 * (get_y_on_line(lower_line, zz_point.pidx) + get_y_on_line(upper_line, zz_point.pidx))
                        for zz_point in zz_points
                    ],
                    mode="markers",
                    hoverinfo="skip",
                    showlegend=True,
                    marker_symbol="x-thin-open",
                    marker_color="white",
                    marker_line_width=2,
                    marker_size=5,
                    name="trend_line_{}".format(i),
                    legendgroup="trend_line",
                )
            )

        for i, order in enumerate(self.orders_closed):
            trend_line = order["desc"]["trend_line"]
            pidx = order["desc"]["pidx"]
            trend_line_id = order["desc"]["trend_line_id"]
            lower_line = trend_line.lower_line
            upper_line = trend_line.upper_line
            y_lower = get_y_on_line(lower_line, pidx)
            y_upper = get_y_on_line(upper_line, pidx)

            zz_points = trend_line.zz_points
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.iloc[lower_line[0][0]]["Open time"],
                        df.iloc[lower_line[1][0]]["Open time"],
                        df.iloc[upper_line[1][0]]["Open time"],
                        df.iloc[upper_line[0][0]]["Open time"],
                        df.iloc[lower_line[0][0]]["Open time"],
                    ],
                    y=[
                        lower_line[0][1],
                        lower_line[1][1],
                        upper_line[1][1],
                        upper_line[0][1],
                        lower_line[0][1],
                    ],
                    mode="lines",
                    line=dict(color="rgba(0, 0, 255, 1)", width=1),
                    fill="toself",
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup="order_{}".format(order.order_id),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.iloc[lower_line[1][0]]["Open time"],
                        df.iloc[pidx]["Open time"],
                        df.iloc[pidx]["Open time"],
                        df.iloc[upper_line[1][0]]["Open time"],
                    ],
                    y=[lower_line[1][1], y_lower, y_upper, upper_line[1][1]],
                    mode="lines",
                    line=dict(color="rgba(243, 156, 18, 0.7)", width=1),
                    showlegend=False,
                    legendgroup="order_{}".format(order.order_id),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_point.pidx]["Open time"] for zz_point in zz_points],
                    y=[
                        0.5 * (get_y_on_line(lower_line, zz_point.pidx) + get_y_on_line(upper_line, zz_point.pidx))
                        for zz_point in zz_points
                    ],
                    mode="markers",
                    hoverinfo="skip",
                    showlegend=False,
                    marker_symbol="x-thin-open",
                    marker_color="yellow",
                    marker_line_width=2,
                    marker_size=5,
                    legendgroup="order_{}".format(order.order_id),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[zz_points[-1].pidx]["Open time"]],
                    y=[
                        0.5
                        * (
                            get_y_on_line(lower_line, zz_points[-1].pidx)
                            + get_y_on_line(upper_line, zz_points[-1].pidx)
                        )
                    ],
                    mode="markers+text",
                    text="{}".format(trend_line_id),
                    textfont=dict(color="white", size=18),
                    showlegend=False,
                    legendgroup="order_{}".format(order.order_id),
                )
            )

        colors = [
            "rgb(242, 54, 69)" if kline["Open"] > kline["Close"] else "rgb(8, 153, 129)" for i, kline in df.iterrows()
        ]
        fig.add_trace(
            go.Bar(x=list(range(len(df))), y=df["Volume"], marker_color=colors, marker_line_width=0, name="Volume"),
            row=2,
            col=1,
        )

        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
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
