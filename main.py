import os
import time
from datetime import datetime
import argparse
import logging
import logging.config
from trade_engine import TradeEngine
from backtest import BackTest
from utils import datetime_to_filename


def config_logging(exchange):
    if not os.path.isdir(os.path.join(os.environ["LOG_DIR"], exchange)):
        os.makedirs(os.path.join(os.environ["LOG_DIR"], exchange), exist_ok=True)
    curr_time = datetime.now()
    logging.config.fileConfig(
        "logging_config.ini",
        defaults={"logfilename": "logs/{}/bot_{}.log".format(exchange, datetime_to_filename(curr_time))},
    )
    logging.getLogger().setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monn auto trading bot")
    parser.add_argument("--mode", required=True, type=str, choices=["live", "test"])
    parser.add_argument("--exch", required=True, type=str)
    parser.add_argument("--exch_cfg_file", required=True, type=str)
    parser.add_argument("--sym_cfg_file", required=True, type=str)
    parser.add_argument("--data_dir", required=False, type=str)
    args = parser.parse_args()

    os.environ["DEBUG_DIR"] = "debug"
    os.environ["LOG_DIR"] = "logs"
    config_logging(args.exch)

    if not os.path.isdir(os.environ["DEBUG_DIR"]):
        os.mkdir(os.environ["DEBUG_DIR"])

    if args.mode == "live":
        trade_engine = TradeEngine(args.exch, args.exch_cfg_file, args.sym_cfg_file)
        if trade_engine.init():
            trade_engine.start()
            try:
                while True:
                    time.sleep(1)
            except (KeyboardInterrupt, SystemExit):
                trade_engine.stop()
                trade_engine.summary_trade_result()
                trade_engine.log_all_trades()
                time.sleep(3)  # Wait for exchange return income
                trade_engine.log_income_history()
    elif args.mode == "test":
        start_time = time.time()
        backtest_engine = BackTest(args.exch, args.sym_cfg_file, args.data_dir)
        backtest_engine.start()
        backtest_engine.summary_trade_result()
        backtest_engine.stop()
        end_time = time.time()
        print("|--------------------------------")
        print(" Backtest finished, time: {:.4f}".format(end_time - start_time))
        print("|--------------------------------")