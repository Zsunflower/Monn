from datetime import datetime
from tabulate import tabulate
import numpy as np
from scipy.optimize import linprog


tf_cron = {
    "4h": {"minute": [0], "hour": [3, 7, 11, 15, 19, 23]},
    "2h": {"minute": [0], "hour": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]},
    "1h": {"minute": [0]},
    "30m": {"minute": [0, 30]},
    "15m": {"minute": [0, 15, 30, 45]},
    "5m": {"minute": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]},
    "3m": {
        "minute": [
            0,
            3,
            6,
            9,
            12,
            15,
            18,
            21,
            24,
            27,
            30,
            33,
            36,
            39,
            42,
            45,
            48,
            51,
            54,
            57,
        ]
    },
    "1m": {},
}

NUM_KLINE_INIT = 300
CANDLE_COLUMNS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]


def datetime_to_filename(dt):
    return str(dt).replace(" ", "-").replace(":", "-")


def timestamp_to_datetime(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda timestamp: datetime.fromtimestamp(timestamp / 1000.0))


def get_pretty_table(df, title_name, transpose=False, tran_col=None):
    if transpose:
        df_T = df.drop([tran_col], axis=1).T
        df_T.columns = df[tran_col]
        df = df_T
    table_stats = tabulate(df, headers="keys", tablefmt="fancy_grid")
    title = "╒" + " {} ".format(title_name).center(len(table_stats.splitlines()[0]) - 2, "═") + "╕"
    table_stats_split = table_stats.splitlines()
    table_stats_split[0] = title
    return "\n" + "\n".join(table_stats_split)


def get_line_coffs(a, c):
    # calculate the slope and y-intercept of the line passing through a and c
    slope = (c[1] - a[1]) / (c[0] - a[0])
    intercept = a[1] - slope * a[0]
    return slope, intercept


# ----------------------------------------------------------------------------------------#
def parse_line_coffs(points):
    x0 = points[0][0]
    xn = points[-1][0]
    X = [
        [
            (point[0] - x0) / (xn - x0),
            (xn - point[0]) / (xn - x0),
        ]
        for point in points
    ]
    Y = [point[1] for point in points]
    return np.array(X), np.array(Y)


def get_y_on_line(line, xi):
    # line (x0, y0), (xn, yn)
    # x0/xi/xn: datetime
    (x0, y0), (xn, yn) = line
    yi = ((xi - x0) / (xn - x0)) * yn + ((xn - xi) / (xn - x0)) * y0
    return yi


def find_uptrend_line(poke_points):
    # poke_points: list (xi, yi)
    # return: ((x0d, y0d), (xnd, ynd))

    X, Y = parse_line_coffs(poke_points)
    obj = [-X[:, 0].sum(), -X[:, 1].sum()]
    lhs_ineq = X
    rhs_ineq = Y
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, method="highs")
    yn, y0 = opt.x
    return ((poke_points[0][0], y0), (poke_points[-1][0], yn))


def is_cross_line(line, kline):
    # line: ((x0d, y0d), (xnd, ynd))
    ykline = get_y_on_line(line, kline["Open time"])
    return (line[0][1] - ykline) * (line[1][1] - ykline) < 0


def find_downtrend_line(peak_points):
    # peak_points: list (xi, yi)
    # return: ((x0u, y0u), (xnu, ynu))

    X, Y = parse_line_coffs(peak_points)
    obj = [X[:, 0].sum(), X[:, 1].sum()]
    lhs_ineq = -X
    rhs_ineq = -Y
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, method="highs")
    yn, y0 = opt.x
    return ((peak_points[0][0], y0), (peak_points[-1][0], yn))


# ----------------------------------------------------------------------------------------#
