import os
import argparse
from datetime import datetime, timedelta
import logging
import logging.config
import json
import itertools
import tempfile
import zipfile
import shutil
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
from typing import List
import pandas as pd
from trader import Trader
from utils import tf_cron, NUM_KLINE_INIT, CANDLE_COLUMNS
from utils import get_pretty_table, datetime_to_filename


def get_combination(params):
    keys = []
    values = []
    for k, v in params.items():
        keys.append(k)
        values.append(v)
    combinations = list(itertools.product(*values))
    return keys, combinations

bot_logger = logging.getLogger("bot_logger")


class Tuning:
    def __init__(self, symbols_trading_cfg_file, data_dir):
        self.data_dir = data_dir
        self.bot_traders: List[Trader] = []
        self.symbols_trading_cfg_file = symbols_trading_cfg_file
        self.debug_dir = os.environ["DEBUG_DIR"]
        self.temp_dir = tempfile.mkdtemp()

    def load_mt5_klines_monthly_data(self, symbol, interval, month, year):
        csv_data_path = os.path.join(self.data_dir, "{}-{}-{}-{:02d}.csv".format(symbol, interval, year, month))
        df = pd.read_csv(csv_data_path)
        df["Open time"] = pd.to_datetime(df["Open time"])
        return df

    def load_klines_monthly_data(self, symbol, interval, month, year):
        return self.load_mt5_klines_monthly_data(symbol, interval, month, year)

    def backtest_bot_trader(self, symbol_cfg):
        bot_trader = Trader(symbol_cfg)
        bot_trader.init_strategies()
        tfs_chart = {}
        # Load kline data for all required timeframes
        for tf in bot_trader.get_required_tfs():
            chart_df = pd.concat(
                [
                    self.load_klines_monthly_data(symbol_cfg["symbol"], tf, month, symbol_cfg["year"])
                    for month in sorted(symbol_cfg["months"])
                ],
                ignore_index=True,
            )
            tfs_chart[tf] = chart_df
        max_time = max([tf_chart.iloc[NUM_KLINE_INIT - 1]["Open time"] for tf_chart in tfs_chart.values()])
        end_time = max([tf_chart.iloc[-1]["Open time"] for tf_chart in tfs_chart.values()])
        tfs_chart_init = {}
        for tf, tf_chart in tfs_chart.items():
            tfs_chart_init[tf] = tf_chart[tf_chart["Open time"] <= max_time][-NUM_KLINE_INIT:]
            tfs_chart[tf] = tf_chart[tf_chart["Open time"] > max_time]
        bot_trader.init_chart(tfs_chart_init)
        bot_trader.attach_oms(None)  # for backtesting don't need oms

        timer = max_time
        end_time = end_time
        bot_logger.info("Start timer from: {} to {}".format(timer, end_time))
        required_tfs = [tf for tf in tf_cron.keys() if tf in bot_trader.get_required_tfs()]

        while timer <= end_time:
            timer += timedelta(seconds=60)
            hour, minute = timer.hour, timer.minute
            for tf in required_tfs:
                cron_time = tf_cron[tf]
                if ("hour" not in cron_time or hour in cron_time["hour"]) and (
                    "minute" not in cron_time or minute in cron_time["minute"]
                ):
                    last_kline = tfs_chart[tf][:1]
                    tfs_chart[tf] = tfs_chart[tf][1:]
                    bot_trader.on_kline(tf, last_kline)
        return bot_trader

    def start(self):
        with open(self.symbols_trading_cfg_file) as f:
            symbols_config = json.load(f)

        def split_list(lst, n):
            """Split a list into n equal segments"""
            k, m = divmod(len(lst), n)
            return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        bot_logger.info("   [+] Start tuning ...")
        args = []
        for symbol_cfg in symbols_config:
            symbols = symbol_cfg["symbols"]
            params_cb = get_combination(symbol_cfg["params"])
            bot_logger.info("   [+] Tuning symbols: {}, total: {} combinations".format(symbols, len(params_cb[1])))
            params_cb = [dict(zip(params_cb[0], params)) for params in params_cb[1]]
            for symbol in symbols:
                sb_cfg = {"symbol": symbol}
                for k, v in symbol_cfg.items():
                    if k not in ["tfs", "params", "symbols", "name"]:
                        sb_cfg[k] = v
                strategies = [{"name": symbol_cfg["name"], "params": param, "tfs": symbol_cfg["tfs"], 
                               "max_sl_pct": symbol_cfg["max_sl_pct"], 
                               "volume": symbol_cfg["volume"]} for param in params_cb]
                # Split list of strategies into list of 10 elements sublist
                sublist_strategies = split_list(strategies, len(strategies) // 10 + 1)
                for sublist_strategy in sublist_strategies:
                    sb_cfg_tpl = {k: v for k, v in sb_cfg.items()}
                    sb_cfg_tpl["strategies"] = sublist_strategy
                    args.append(sb_cfg_tpl)
        bot_logger.info("   [+] Run total {} trials".format(len(args)))
        # preload data
        for symbol_cfg in symbols_config:
            symbols = symbol_cfg["symbols"]
            for symbol in symbols:
                for tf in symbol_cfg["tfs"].values():
                    for mnt in symbol_cfg["months"]:
                        self.load_klines_monthly_data(symbol, tf, mnt, symbol_cfg["year"])

        with Pool(cpu_count()) as pool:
            self.bot_traders.extend(pool.map(self.backtest_bot_trader, args))
        bot_logger.info("   [*] Tuning finished")

    def summary_trade_result(self):
        final_backtest_stats = []
        for bot_trader in self.bot_traders:
            bot_trader.close_opening_orders()
            backtest_stats = bot_trader.statistic_trade()
            backtest_stats.insert(loc=0, column="SYMBOL", value=bot_trader.get_symbol_name())
            backtest_stats = backtest_stats.iloc[:-1]
            backtest_stats.insert(loc=2, column="params", value=bot_trader.get_strategy_params())
            final_backtest_stats.append(backtest_stats)
        table_stats = pd.concat(final_backtest_stats, axis=0, ignore_index=True)
        return table_stats

    def stop(self):
        shutil.rmtree(self.temp_dir)


def config_logging(exchange):
    curr_time = datetime.now()
    logging.config.fileConfig(
        "logging_config.ini",
        defaults={"logfilename": "logs/{}/bot_tuning_{}.log".format(exchange, datetime_to_filename(curr_time))},
    )
    logging.getLogger().setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monn auto trading bot")
    parser.add_argument("--sym_cfg_file", required=True, type=str)
    parser.add_argument("--data_dir", required=False, type=str)
    args = parser.parse_args()

    config_logging("binance")
    os.environ["DEBUG_DIR"] = "debug"


    tun_engine = Tuning(args.sym_cfg_file, args.data_dir)
    tun_engine.start()
    table_stats = tun_engine.summary_trade_result()
    table_stats.to_csv(os.path.splitext(args.sym_cfg_file)[0] + ".csv")
    tun_engine.stop()