import os
import time
from datetime import datetime
import logging
import json
from typing import List
import pandas as pd
import tzlocal
from apscheduler.schedulers.background import BackgroundScheduler
from trader import Trader
from exchange_loader import ExchangeLoader, OMSLoader
from utils import tf_cron, NUM_KLINE_INIT
from utils import get_pretty_table


bot_logger = logging.getLogger("bot_logger")


class TradeEngine:
    def __init__(self, exchange_name, exc_cfg_file, symbols_trading_cfg_file):
        self.exchange_name = exchange_name
        self.exc_cfg_file = exc_cfg_file
        self.symbols_trading_cfg_file = symbols_trading_cfg_file
        self.sched = BackgroundScheduler(timezone=str(tzlocal.get_localzone()))
        self.required_tfs = []
        self.last_updated_tfs = {}
        self.bot_traders: List[Trader] = []

    def init(self):
        self.mt5_api = ExchangeLoader(self.exc_cfg_file).get_exchange(exchange_name=self.exchange_name)
        if self.mt5_api.initialize() and self.mt5_api.login():
            bot_logger.info("[+] Init MT5 API success")
        else:
            bot_logger.info("[-] Init MT5 API failed, stop")
            return False
        self.oms = OMSLoader().get_oms(self.exchange_name, self.mt5_api)
        self.init_bot_traders(self.symbols_trading_cfg_file)
        return True

    def init_bot_traders(self, symbols_trading_cfg_file):
        with open(symbols_trading_cfg_file) as f:
            symbols_config = json.load(f)
        for symbol_cfg in symbols_config:
            bot_logger.info("[+] Create trading bot for symbol: {}".format(symbol_cfg["symbol"]))
            bot_trader = Trader(symbol_cfg)
            bot_trader.init_strategies()
            tfs_chart = {}
            for tf in bot_trader.get_required_tfs():
                chart_df = self.mt5_api.klines(symbol_cfg["symbol"], tf, limit=NUM_KLINE_INIT + 1)
                chart_df = chart_df[:-1]
                tfs_chart[tf] = chart_df
                bot_logger.info(
                    "[+] Init bot symbol: {}, tf: {}, from: {} to: {}".format(
                        symbol_cfg["symbol"], tf, chart_df.iloc[0]["Open time"], chart_df.iloc[-1]["Open time"]
                    )
                )
                self.last_updated_tfs[tf] = chart_df.iloc[-1]["Open time"]
            bot_trader.init_chart(tfs_chart)
            bot_trader.attach_oms(self.oms)
            self.required_tfs.extend(bot_trader.get_required_tfs())
            self.bot_traders.append(bot_trader)
        self.required_tfs = set(self.required_tfs)
        # Sort timeframes from large to small
        self.required_tfs = [tf for tf in tf_cron.keys() if tf in self.required_tfs]
        bot_logger.info("[+] Required timeframes: {}".format(self.required_tfs))
        bot_logger.info("[+] Last updated timeframes: {}".format(self.last_updated_tfs))

    def __update_next_kline__(self):
        curr_time = datetime.now()
        hour, minute = curr_time.hour, curr_time.minute
        time_retry = 0
        for tf in self.required_tfs:
            cron_time = tf_cron[tf]
            if ("hour" not in cron_time or hour in cron_time["hour"]) and (
                "minute" not in cron_time or minute in cron_time["minute"]
            ):
                bot_logger.info("   [+] Update tf: {}, time: {}".format(tf, curr_time))
                last_updated_tf = self.last_updated_tfs[tf]
                for bot_trader in self.bot_traders:
                    if tf in bot_trader.get_required_tfs():
                        while True:
                            chart_df = self.mt5_api.klines(bot_trader.get_symbol_name(), tf, limit=2)
                            last_kline = chart_df[:1]
                            if last_kline.iloc[0]["Open time"] > self.last_updated_tfs[tf]:
                                break
                            time_retry += 0.1
                            time.sleep(0.1)
                            if time_retry > 10:
                                bot_logger.info("   [+] Update next kline timeout: {}".format(curr_time))
                                break
                        if last_kline.iloc[0]["Open time"] > self.last_updated_tfs[tf]:
                            bot_logger.info(
                                "       [+] {}, kline: {}".format(
                                    bot_trader.get_symbol_name(), last_kline.iloc[0]["Open time"]
                                )
                            )
                            bot_trader.on_kline(tf, last_kline)
                            last_updated_tf = last_kline.iloc[0]["Open time"]
                self.last_updated_tfs[tf] = last_updated_tf

    def __oms_loop__(self):
        self.oms.monitor_trades()

    def start(self):
        bot_logger.info("[*] Start trading bot, time: {}".format(datetime.now()))
        self.sched.add_job(
            self.__update_next_kline__,
            "cron",
            args=[],
            max_instances=1,
            replace_existing=False,
            minute="0-59",
            second=1,
        )
        self.sched.add_job(self.__oms_loop__, "interval", seconds=15)
        self.sched.start()

    def stop(self):
        bot_logger.info("[*] Stop trading bot, time: {}".format(datetime.now()))
        self.oms.close_all_trade()
        self.sched.shutdown(wait=True)

    def summary_trade_result(self):
        final_backtest_stats = []
        for bot_trader in self.bot_traders:
            bot_trader.close_opening_orders()
            backtest_stats = bot_trader.statistic_trade()
            bot_logger.info(get_pretty_table(backtest_stats, bot_trader.get_symbol_name(), transpose=True, tran_col="NAME"))
            backtest_stats.loc[len(backtest_stats) - 1, "NAME"] = bot_trader.get_symbol_name()
            summary = backtest_stats.loc[len(backtest_stats) - 1 :]
            final_backtest_stats.append(summary)

            bot_trader.log_orders()
            bot_trader.plot_strategy_orders()

        table_stats = pd.concat(final_backtest_stats, axis=0, ignore_index=True)
        s = table_stats.sum(axis=0)
        table_stats.loc[len(table_stats)] = s
        table_stats.loc[len(table_stats) - 1, "NAME"] = "TOTAL"
        bot_logger.info(get_pretty_table(table_stats, "SUMMARY", transpose=True, tran_col="NAME"))
        return table_stats

    def log_all_trades(self):
        closed_trades, active_trades = self.oms.get_trades()
        df_closed = pd.DataFrame([trade.__to_dict__() for trade in closed_trades.values()])
        df_closed.to_csv(os.path.join(os.environ["DEBUG_DIR"], "closed_trades.csv"))
        df_active = pd.DataFrame([trade.__to_dict__() for trade in active_trades.values()])
        df_active.to_csv(os.path.join(os.environ["DEBUG_DIR"], "active_trades.csv"))

    def log_income_history(self):
        df_ic = self.oms.get_income_history()
        if len(df_ic) > 0:
            bot_logger.info(get_pretty_table(df_ic, "INCOME SUMMARY"))
        df_ic.to_csv(os.path.join(os.environ["DEBUG_DIR"], "income_summary.csv"))
