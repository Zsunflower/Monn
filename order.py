from enum import Enum


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    HIT_SL = "HIT_SL"
    HIT_TP = "HIT_TP"
    STOPPED = "STOPPED"


class OrderTemplate:
    def __init__(self, symbol, quantity, entry, order_side: OrderSide, order_type: OrderType):
        self.symbol = symbol
        self.quantity = quantity
        self.entry = entry
        self.order_type = order_type
        self.order_side = order_side
        if self.order_side == OrderSide.BUY:
            self.position_side = PositionSide.LONG
        else:
            self.position_side = PositionSide.SHORT

    def get_order_params(self, **kwargs):
        params = {
            "symbol": self.symbol,
            "type": self.order_type.value,
            "side": self.order_side.value,
            "positionSide": self.position_side.value,
            "quantity": self.quantity,
        }
        params.update(kwargs)
        return params

    def get_main_order(self):
        return (
            self.get_order_params()
            if self.order_type == OrderType.MARKET
            else self.get_order_params(timeInForce="GTC", price=self.entry)
        )

    def get_close_order(self):
        return self.get_order_params(
            type=OrderType.MARKET.value,
            side=OrderSide.BUY.value if self.order_side == OrderSide.SELL else OrderSide.SELL.value,
        )

    def get_tp_order(self, tp):
        return self.get_order_params(
            type="TAKE_PROFIT_MARKET",
            side=OrderSide.BUY.value if self.order_side == OrderSide.SELL else OrderSide.SELL.value,
            stopPrice=tp,
            workingType="MARK_PRICE",
        )

    def get_sl_order(self, sl):
        return self.get_order_params(
            type="STOP_MARKET",
            side=OrderSide.BUY.value if self.order_side == OrderSide.SELL else OrderSide.SELL.value,
            stopPrice=sl,
            workingType="MARK_PRICE",
        )


class Order:
    #
    #      |-------------| TP
    #      |             |
    #      |             |
    #      |     BUY     |
    #      |             |
    #      |-------------| ENTRY
    #      |             |
    #      |-------------| SL
    # =================================#
    #      |-------------| SL
    #      |             |
    #      |-------------| ENTRY
    #      |             |
    #      |             |
    #      |    SELL     |
    #      |             |
    #      |-------------| TP
    #
    __order_id__ = 1

    def __init__(
        self,
        order_type: OrderType,
        order_side: OrderSide,
        entry,
        tp=None,
        sl=None,
        status=OrderStatus.PENDING,
    ):
        # type="LIMIT"/"MARKET",
        # side="BUY"/"SELL",
        self.order_id = Order.__order_id__
        Order.__order_id__ += 1
        self.type = order_type
        self.side = order_side
        self.entry = entry
        self.tp = tp
        self.sl = sl
        self.status = status
        self.rr = 0
        self.__attrs__ = {}
        self.calc_stats()

    def calc_stats(self):
        self.reward_ratio = None
        self.risk_ratio = None
        if self.tp:
            self.reward_ratio = round(abs(self.tp - self.entry) / self.entry, 4)
        if self.sl:
            self.risk_ratio = round(abs(self.sl - self.entry) / self.entry, 4)
        if self.reward_ratio and self.risk_ratio:
            self.rr = round(self.reward_ratio / self.risk_ratio, 4)

    def is_valid(self):
        if self.side == OrderSide.BUY:
            if self.has_tp() and self.tp < self.entry:
                return False
            if self.has_sl() and self.sl > self.entry:
                return False
            return True
        if self.has_tp() and self.tp > self.entry:
            return False
        if self.has_sl() and self.sl < self.entry:
            return False
        return True

    def has_sl(self):
        return self.sl is not None

    def has_tp(self):
        return self.tp is not None

    def adjust_tp(self, tp):
        self.tp = tp
        self.calc_stats()

    def adjust_sl(self, sl):
        self.sl = sl
        self.calc_stats()
        if (self.side == OrderSide.SELL and self.sl <= self.entry) or (
            self.side == OrderSide.BUY and self.sl >= self.entry
        ):
            self.risk_ratio = 0

    def adjust_entry(self, entry):
        self.entry = entry
        self.calc_stats()

    def update_status(self, kline):
        open_time = kline["Open time"]
        ohlc = kline[["Open", "High", "Low", "Close"]]
        if self.status == OrderStatus.PENDING:
            # limit order
            if (ohlc["High"] - self.entry) * (ohlc["Low"] - self.entry) <= 0:
                self.status = OrderStatus.FILLED
                self.__attrs__["FILL_TIME"] = open_time
            if (
                self.status == OrderStatus.FILLED
                and self.has_sl()
                and (ohlc["High"] - self.sl) * (ohlc["Low"] - self.sl) <= 0
            ):
                # order hit sl
                self.status = OrderStatus.HIT_SL
                self.__attrs__["STOP_TIME"] = open_time
            if (
                self.status == OrderStatus.FILLED
                and self.has_tp()
                and (ohlc["High"] - self.tp) * (ohlc["Low"] - self.tp) <= 0
            ):
                # order hit tp
                self.status = OrderStatus.HIT_TP
                self.__attrs__["STOP_TIME"] = open_time
        elif self.status == OrderStatus.FILLED:
            if (
                self.has_sl()
                and (ohlc["High"] - self.sl) * (ohlc["Low"] - self.sl) <= 0
                or (self.side == OrderSide.SELL and ohlc["High"] >= self.sl)
                or (self.side == OrderSide.BUY and ohlc["Low"] <= self.sl)
            ):
                # order hit sl
                self.status = OrderStatus.HIT_SL
                self.__attrs__["STOP_TIME"] = open_time
            elif self.has_tp() and (ohlc["High"] - self.tp) * (ohlc["Low"] - self.tp) <= 0:
                # order hit tp
                self.status = OrderStatus.HIT_TP
                self.__attrs__["STOP_TIME"] = open_time

    def close(self, kline):
        if self.status == OrderStatus.FILLED:
            self.status = OrderStatus.STOPPED
            self.__attrs__["STOP_TIME"] = kline["Open time"]
            self.__attrs__["STOP_PRICE"] = kline["Close"]
            change_pc = round((kline["Close"] - self.entry) / self.entry, 4)
            self.__attrs__["PnL"] = change_pc if self.side == OrderSide.BUY else -change_pc

    def is_closed(self):
        return self.status in [OrderStatus.HIT_SL, OrderStatus.HIT_TP, OrderStatus.STOPPED]

    def get_PnL(self):
        if self.status == OrderStatus.HIT_SL:
            pnl = round(abs(self.sl - self.entry) / self.entry, 4)
            if (self.side == OrderSide.SELL and self.sl <= self.entry) or (
                self.side == OrderSide.BUY and self.sl >= self.entry
            ):
                return pnl
            return -pnl
        if self.status == OrderStatus.HIT_TP:
            return self.reward_ratio
        if self.status == OrderStatus.STOPPED:
            return self.__attrs__["PnL"]
        return 0

    def __getitem__(self, __name: str):
        return self.__attrs__[__name]

    def __setitem__(self, __name, __value):
        self.__attrs__[__name] = __value

    def __contains__(self, key):
        return key in self.__attrs__

    def __to_dict__(self):
        return {
            "order_id": self.order_id,
            "type": self.type.value,
            "side": self.side.value,
            "entry": self.entry,
            "tp": self.tp,
            "sl": self.sl,
            "status": self.status.value,
            "reward_ratio": self.reward_ratio,
            "risk_ratio": self.risk_ratio,
            "rr": self.rr,
            "attrs": self.__attrs__,
        }

    def __str__(self) -> str:
        return self.__to_dict__().__str__()
