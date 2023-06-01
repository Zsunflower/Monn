from order import Order


class Trade:
    __trade_id__ = 10000

    def __init__(self, order: Order, volume):
        self.trade_id = Trade.__trade_id__
        Trade.__trade_id__ += 1
        self.order = order
        self.volume = volume
        self.main_order_params = None
        self.close_order_params = None
        self.main_order = None
        self.close_order = None
        self.trace = []

    def __to_dict__(self):
        trade_dict = self.order.__to_dict__()
        trade_dict["volume"] = self.volume
        trade_dict["main_order_params"] = self.main_order_params
        trade_dict["close_order_params"] = self.close_order_params
        trade_dict["main_order"] = self.main_order
        trade_dict["close_order"] = self.close_order
        trade_dict["trace"] = self.trace
        return trade_dict
