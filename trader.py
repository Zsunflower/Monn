import os
import logging
import pandas as pd
from strategy_utils import load_strategy
from order import Order, OrderSide, OrderStatus, OrderType

bot_logger = logging.getLogger("bot_logger")


class Trader:
    # Trading bot for single symbol
    ### Example ###
    # {
    #     "symbol": "XAUUSDm",
    #     "strategies": [
    #         {
    #             "name": "break_strategy",
    #             "params": {
    #                 "min_num_cuml": 10,
    #                 "min_zz_pct": 0.5,
    #                 "zz_dev": 2,
    #                 "ma_vol": 25,
    #                 "vol_ratio_ma": 1.8,
    #                 "kline_body_ratio": 2,
    #                 "sl_fix_mode": "ADJ_SL"
    #             },
    #             "tfs": {
    #                 "tf": "15m"
    #             },
    #             "max_sl_pct": 1, # max sl percent(example: sl will not smaller than (1 - max_Sl_pct / 100) * entry in buy order), 0 for infinity sl
    #             "volume": 0.01
    #         }
    #     ],
    #     "months": [
    #         1
    #     ],
    #     "year": 2023
    # }

    def __init__(self, json_cfg):
        self.json_cfg = json_cfg
        self.symbol_name = self.json_cfg["symbol"]
        self.required_tfs = {}
        self.strategies = []
        self.log_dir = os.path.join(os.environ["DEBUG_DIR"], self.symbol_name)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def init_chart(self, tfs_chart):
        # tfs_chart: {"1h": chart_1h, "15m": chart_15m}
        self.tfs_chart = tfs_chart
        for strategy in self.strategies:
            strategy.attach(self.tfs_chart)
            strategy.attach_trader(self)

    def init_strategies(self):
        for strategy_def in self.json_cfg["strategies"]:
            strategy = load_strategy(strategy_def)
            if strategy.is_params_valid():
                bot_logger.info(
                    "   [+] Load strategy: {}, params {} success".format(strategy_def["name"], strategy_def["params"])
                )
                strategy.set_volume(strategy_def["volume"])
                strategy.set_max_sl_pct(strategy_def.get("max_sl_pct"))
                self.strategies.append(strategy)
                for tf in strategy_def["tfs"].values():
                    if tf in self.required_tfs:
                        self.required_tfs[tf].append(strategy)
                    else:
                        self.required_tfs[tf] = [strategy]
            else:
                bot_logger.error(
                    "   [-] Load strategy: {}, params {} failed, invalid params".format(
                        strategy_def["name"], strategy_def["params"]
                    )
                )

    def attach_oms(self, oms):
        self.oms = oms

    def create_trade(self, order: Order, volume):
        if self.oms:
            bot_logger.info(
                "   [+] Create new order, symbol: {}, strategy: {}".format(self.symbol_name, order["description"])
            )
            bot_logger.info("    - {}".format(order))
            order["symbol"] = self.symbol_name
            order["trade_id"] = self.oms.create_trade(order, volume)

    def close_trade(self, order: Order):
        if self.oms:
            if "trade_id" in order:
                self.oms.close_trade(order["trade_id"])

    def adjust_sl(self, order: Order, sl):
        if self.oms:
            if "trade_id" in order:
                self.oms.adjust_sl(order["trade_id"], sl)

    def get_required_tfs(self):
        return list(self.required_tfs.keys())

    def get_symbol_name(self):
        return self.symbol_name

    def statistic_trade(self):
        # statistic closed trades for each strategy
        stats = []
        for strategy in self.strategies:
            stats.append(strategy.summary_PnL())
        df_stats = pd.DataFrame(stats)
        s = df_stats.sum(axis=0)
        s["AVG(%)"] = s["TOTAL_PnL(%)"] / s["TOTAL"] if s["TOTAL"] > 0 else 0
        df_stats.loc[len(df_stats)] = s
        df_stats.loc[len(df_stats) - 1, "NAME"] = "TOTAL"
        # bot_logger.info("\n" + 30 * "-" + " {} ".format(self.symbol_name) + 30 * "-" + "\n" + df_stats.to_string() + "\n" + 70 * "-")
        return df_stats

    def close_opening_orders(self):
        for strategy in self.strategies:
            strategy.close_opening_orders()

    def get_strategy_params(self):
        return [strategy.params for strategy in self.strategies]

    def fix_order(self, order: Order, sl_fix_mode, max_sl_pct):
        # set tp/sl follow config(max sl, min tp, min rr, sl_fix_mode)
        if max_sl_pct:
            # max_sl_pct was set
            max_sl_pct = max_sl_pct / 100
            max_sl = (1 - max_sl_pct) * order.entry if order.side == OrderSide.BUY else (1 + max_sl_pct) * order.entry
            if not order.has_sl():
                order.adjust_sl(max_sl)
                return order
            if (order.side == OrderSide.BUY and order.sl >= max_sl) or (
                order.side == OrderSide.SELL and order.sl <= max_sl
            ):
                return order
            if sl_fix_mode == "ADJ_SL":
                order.adjust_sl(max_sl)
                return order
            elif sl_fix_mode == "IGNORE":
                bot_logger.info("   [-] IGNORE order: {}".format(order))
                return None
            elif sl_fix_mode == "ADJ_ENTRY":
                adjusted_entry = (
                    order.sl / (1 - max_sl_pct) if order.side == OrderSide.BUY else order.sl / (1 + max_sl_pct)
                )
                order.adjust_entry(adjusted_entry)
                order.status = OrderStatus.PENDING
                order.type = OrderType.LIMIT
                return order
        return order

    def log_orders(self):
        df_orders = []
        for strategy in self.strategies:
            df = pd.DataFrame([order.__to_dict__() for order in strategy.orders_closed])
            df_orders.append(df)
        df_orders = pd.concat(df_orders)
        df_orders.to_csv(os.path.join(self.log_dir, "{}_orders.csv".format(self.symbol_name)))

    def plot_strategy_orders(self):
        for strategy in self.strategies:
            fig = strategy.plot_orders()
            fig.write_html(
                os.path.join(self.log_dir, "{}.html".format(strategy.get_name())),
                include_plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
            )

    def on_kline(self, tf, kline):
        # update strategies
        self.tfs_chart[tf] = pd.concat([self.tfs_chart[tf], kline], ignore_index=True)
        for strategy in self.required_tfs[tf]:
            strategy.update(tf)
