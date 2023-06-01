from datetime import datetime
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from order import Order, OrderSide, OrderStatus


class BaseStrategy:
    def __init__(self, name, params, tfs):
        self.name = name
        self.params = params
        self.tfs = tfs
        # orders in activate created by the strategy
        self.orders_opening: List[Order] = []
        # orders closed created by the strategy
        self.orders_closed: List[Order] = []
        self.start_trading_time = None
        self.set_description()

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def set_description(self):
        self.description = self.name
        for tfn, tf in self.tfs.items():
            self.description += "_{}_{}".format(tfn, tf)

    def set_volume(self, volume):
        self.volume = volume

    def set_max_sl_pct(self, max_sl_pct):
        self.max_sl_pct = max_sl_pct

    def get_name(self):
        return self.name

    def attach_trader(self, trader):
        self.trader = trader

    def init_indicators(self):
        pass

    def update_indicators(self, tf):
        pass

    def check_required_params(self):
        return False

    def is_params_valid(self):
        return False

    def update_orders_status(self, last_kline):
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            order.update_status(last_kline)
            if order.is_closed():
                self.orders_closed.append(order)
                del self.orders_opening[i]

    def close_order_by_side(self, last_kline, order_side: OrderSide):
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            if order.side == order_side:
                order.close(last_kline)
                self.trader.close_trade(order)
                if order.is_closed():
                    self.orders_closed.append(order)
                del self.orders_opening[i]

    def close_opening_orders(self, last_kline):
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            order.close(last_kline)
            self.trader.close_trade(order)
            if order.is_closed():
                self.orders_closed.append(order)
            del self.orders_opening[i]

    def summary_PnL(self):
        num_long_order = 0
        num_short_order = 0
        num_long_profit = 0
        num_short_profit = 0
        long_PnL = 0
        long_tp_profit = 0
        long_sl_loss = 0
        short_PnL = 0
        short_tp_profit = 0
        short_sl_loss = 0

        def get_percent(x, y):
            return x / y * 100 if y > 0 else 0

        for order in self.orders_closed:
            pnl = order.get_PnL()
            if order.side == OrderSide.BUY:
                num_long_order += 1
                long_PnL += pnl
                if pnl > 0:
                    num_long_profit += 1
                    long_tp_profit += pnl
                elif pnl < 0:
                    long_sl_loss += pnl
            else:
                num_short_order += 1
                short_PnL += pnl
                if pnl > 0:
                    num_short_profit += 1
                    short_tp_profit += pnl
                elif pnl < 0:
                    short_sl_loss += pnl
        return {
            "NAME": self.get_name(),
            "LONG": num_long_order,
            "LONG_TP_PnL(%)": 100 * long_tp_profit,
            "LONG_SL_PnL(%)": 100 * long_sl_loss,
            "LONG_PnL(%)": 100 * long_PnL,
            "NUM_LONG_TP": num_long_profit,
            "LONG_TP_PCT": get_percent(num_long_profit, num_long_order),
            "SHORT": num_short_order,
            "SHORT_TP_PnL(%)": 100 * short_tp_profit,
            "SHORT_SL_PnL(%)": 100 * short_sl_loss,
            "SHORT_PnL(%)": 100 * short_PnL,
            "NUM_SHORT_TP": num_short_profit,
            "SHORT_TP_PCT": get_percent(num_short_profit, num_short_order),
            "TOTAL": num_long_order + num_short_order,
            "TOTAL_TP_PCT": get_percent(num_long_profit + num_short_profit, num_long_order + num_short_order),
            "TOTAL_PnL(%)": 100 * (long_PnL + short_PnL),
            "AVG(%)": get_percent(long_PnL + short_PnL, num_long_order + num_short_order),
        }

    def update(self, tf):
        self.update_indicators(tf)
        self.update_orders_status(self.tfs_chart[tf].iloc[-1])

    def plot_orders(self, fig, tf, row, col, dt2idx=None):
        df = self.tfs_chart[tf]
        fig.add_trace(
            go.Candlestick(
                x=df["Open time"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                text=list(dt2idx.keys()) if dt2idx else None,
                increasing={"fillcolor": "rgb(8, 153, 129)"},
                decreasing={"fillcolor": "rgb(242, 54, 69)"},
            ),
            row=row,
            col=col,
        )
        if self.start_trading_time:
            fig.add_shape(
                type="line",
                x0=dt2idx[self.start_trading_time] if dt2idx else self.start_trading_time,
                y0=0,
                x1=dt2idx[self.start_trading_time] if dt2idx else self.start_trading_time,
                y1=1,
                line=dict(width=1, color="orange"),
                yref="y domain",
            )
        for order in self.orders_closed:
            if order.get_PnL() > 0:
                line_color = "rgba(0, 255, 0, 1.0)"
            else:
                line_color = "rgba(255, 0, 0, 1.0)"
            x0 = order.__attrs__["FILL_TIME"] if dt2idx is None else dt2idx[order.__attrs__["FILL_TIME"]]
            x1 = order.__attrs__["STOP_TIME"] if dt2idx is None else dt2idx[order.__attrs__["STOP_TIME"]]
            text = "risk: {}<br>reward: {}<br>rr: {}<br>PnL: {}<br>id: {}".format(
                order.risk_ratio, order.reward_ratio, order.rr, order.get_PnL(), order.order_id
            )
            y = order.entry
            if order.has_sl():
                if order.risk_ratio > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[order.entry, order.entry, order.sl, order.sl, order.entry],
                            mode="lines",
                            line=dict(width=1, color=line_color),
                            fill="toself",
                            fillcolor="rgba(242, 54, 69, 0.2)",
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup="order_{}".format(order.order_id),
                        ),
                        row=row,
                        col=col,
                    )
                elif order.status == OrderStatus.HIT_SL:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[order.entry, order.entry, order.sl, order.sl, order.entry],
                            mode="lines",
                            line=dict(width=1, color=line_color),
                            fill="toself",
                            fillcolor="rgba(0, 0, 200, 0.2)",
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup="order_{}".format(order.order_id),
                        ),
                        row=row,
                        col=col,
                    )
                y = max(y, order.sl)
            if order.has_tp():
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[order.entry, order.entry, order.tp, order.tp, order.entry],
                        mode="lines",
                        line=dict(width=1, color=line_color),
                        fill="toself",
                        fillcolor="rgba(8, 153, 129, 0.2)",
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup="order_{}".format(order.order_id),
                    ),
                    row=row,
                    col=col,
                )
                y = max(y, order.tp)
            if order.status == OrderStatus.STOPPED:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[
                            order.entry,
                            order.entry,
                            order.__attrs__["STOP_PRICE"],
                            order.__attrs__["STOP_PRICE"],
                            order.entry,
                        ],
                        mode="lines",
                        line=dict(width=1, color=line_color),
                        fill="toself",
                        fillcolor="rgba(0, 0, 200, 0.2)",
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup="order_{}".format(order.order_id),
                    ),
                    row=row,
                    col=col,
                )
                y = max(y, order.__attrs__["STOP_PRICE"])
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[order.entry, order.entry],
                    mode="lines",
                    line=dict(width=2, color="rgba(255, 255, 255, 1.0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="order_{}".format(order.order_id),
                    legendgroup="order_{}".format(order.order_id),
                ),
                row=row,
                col=col,
            )
            if "LIMIT_TIME" in order.__attrs__:
                fig.add_trace(
                    go.Scatter(
                        x=[
                            x0,
                            order.__attrs__["LIMIT_TIME"] if dt2idx is None else dt2idx[order.__attrs__["LIMIT_TIME"]],
                        ],
                        y=[order.entry, order.entry],
                        mode="lines",
                        line=dict(width=1, dash="dash", color="rgba(255, 255, 255, 0.7)"),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup="order_{}".format(order.order_id),
                    ),
                    row=row,
                    col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=[x0],
                    y=[y],
                    mode="text",
                    text=[text],
                    textposition="top right",
                    textfont={"color": "rgba(255, 255, 255, 1.0)"},
                    hoverinfo="skip",
                    showlegend=True,
                    name="order_{}".format(order.order_id),
                    legendgroup="order_text",
                ),
                row=row,
                col=col,
            )
