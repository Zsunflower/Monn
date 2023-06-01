import json
from exchange import *


class ExchangeLoader:
    __exchanges__ = {}
    __exchanges_map__ = {
        "mt5": "MT5API",
    }

    def __init__(self, exc_cfg_file):
        with open(exc_cfg_file) as f:
            self.exchange_configs = json.load(f)

    def get_exchange(self, exchange_name):
        if exchange_name not in ExchangeLoader.__exchanges__:
            ExchangeLoader.__exchanges__[exchange_name] = self.__load_exchange__(exchange_name)
        return ExchangeLoader.__exchanges__[exchange_name]

    def __load_exchange__(self, exchange_name):
        exchange_name_config = self.exchange_configs[exchange_name]
        return globals()[ExchangeLoader.__exchanges_map__[exchange_name]](exchange_name_config)


class OMSLoader:
    __oms__ = {}
    __oms_map__ = {
        "mt5": "MT5OMS",
    }

    def get_oms(self, exchange_name, exchange):
        if exchange_name not in OMSLoader.__oms__:
            OMSLoader.__oms__[exchange_name] = self.__load_oms__(exchange_name, exchange)
        return OMSLoader.__oms__[exchange_name]

    def __load_oms__(self, exchange_name, exchange):
        return globals()[OMSLoader.__oms_map__[exchange_name]](exchange)
