import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5

bot_logger = logging.getLogger("bot_logger")


class MT5API:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        # connect to MetaTrader 5
        if not mt5.initialize():
            mt5.shutdown()
            return False
        return True

    def login(self):
        # connect to the trade account specifying a server
        authorized = mt5.login(self.config["account"], server=self.config["server"])
        if authorized:
            bot_logger.info("[+] Login success, account info: ")
            account_info = mt5.account_info()._asdict()
            bot_logger.info(account_info)
            return True
        else:
            bot_logger.info("[-] Login failed, check account infor")
            return False

    def round_price(self, symbol, price):
        symbol_info = mt5.symbol_info(symbol)
        return round(price, symbol_info.digits)

    def get_assets_balance(self, assets=["USD"]):
        assets = {}
        return assets

    def tick_ask_price(self, symbol):
        return mt5.symbol_info_tick(symbol).ask

    def tick_bid_price(self, symbol):
        return mt5.symbol_info_tick(symbol).bid

    def klines(self, symbol: str, interval: str, **kwargs):
        symbol_rates = mt5.copy_rates_from_pos(
            symbol, getattr(mt5, "TIMEFRAME_" + (interval[-1:] + interval[:-1]).upper()), 0, kwargs["limit"]
        )
        df = pd.DataFrame(symbol_rates)
        df["time"] += -time.timezone
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
        df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
        return df

    def place_order(self, params):
        return mt5.order_send(params)

    def history_deals_get(self, position_id):
        return mt5.history_deals_get(position=position_id)
