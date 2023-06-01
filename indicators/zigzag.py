from enum import Enum
from typing import List
import numpy as np


class POINT_TYPE(Enum):
    PEAK_POINT = "PEAK_POINT"
    POKE_POINT = "POKE_POINT"


class TREND_TYPE(Enum):
    UP_TREND = "UP_TREND"
    DOWN_TREND = "DOWN_TREND"
    UNK_TREND = "UNK_TREND"


class SRLine(object):
    # Support/Resistance line
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __to_dict__(self):
        return {"low": self.low, "high": self.high}

    def __repr__(self):
        return str(self.__to_dict__())


class ZZPoint(object):
    def __init__(self, pidx, ptype: POINT_TYPE, pline: SRLine):
        self.pidx = pidx
        self.ptype = ptype
        self.pline = pline

    def __to_dict__(self):
        return {"pidx": self.pidx, "ptype": self.ptype.value, "pline": self.pline}

    def __repr__(self):
        return str(self.__to_dict__())


def merge_break_points(zz_points, min_div, from_idx=0):
    # merge zz_points to wave >= min_div
    i = from_idx
    while i < len(zz_points) - 2:
        ftp = zz_points[i]
        scp = zz_points[i + 1]
        if ftp.ptype == POINT_TYPE.POKE_POINT:
            l = abs(scp.pline.high - ftp.pline.low) / ftp.pline.low
        else:
            l = abs(scp.pline.low - ftp.pline.high) / ftp.pline.high
        if l < min_div:
            # delete 2 zz_points have wave len < min_div
            del zz_points[i + 1]
            del zz_points[i]
            if ftp.ptype == POINT_TYPE.POKE_POINT:
                # merge old zz_point with new zz_point
                if ftp.pline.high < zz_points[i].pline.high:
                    ftp.pline.low = min(ftp.pline.low, zz_points[i].pline.low)
                    zz_points[i] = ftp
                    if i + 1 < len(zz_points):
                        if scp.pline.low > zz_points[i + 1].pline.low:
                            scp.pline.high = max(scp.pline.high, zz_points[i + 1].pline.high)
                            zz_points[i + 1] = scp
                        else:
                            zz_points[i + 1].pline.high = max(scp.pline.high, zz_points[i + 1].pline.high)
                else:
                    zz_points[i].pline.low = min(ftp.pline.low, zz_points[i].pline.low)
            else:
                if ftp.pline.low > zz_points[i].pline.low:
                    ftp.pline.high = max(ftp.pline.high, zz_points[i].pline.high)
                    zz_points[i] = ftp
                    if i + 1 < len(zz_points):
                        if scp.pline.high < zz_points[i + 1].pline.high:
                            scp.pline.low = min(scp.pline.low, zz_points[i + 1].pline.low)
                            zz_points[i + 1] = scp
                        else:
                            zz_points[i + 1].pline.low = min(scp.pline.low, zz_points[i + 1].pline.low)
                else:
                    zz_points[i].pline.high = max(ftp.pline.high, zz_points[i].pline.high)
            i -= 1
        i += 1
    return zz_points


class ZigZag(object):
    def __init__(self, df, kernel_size=3):
        self.df = df
        self.kernel_size = kernel_size
        self.zz_points: List[ZZPoint] = []

        self.conv_col = self.df["Close"] #self.df[["Open", "Close"]].max(axis=1)
        self.find_break_points()
        self.fix_break_points()

    def find_break_points(self):
        # |-------------------------------|
        #    |------------------------|
        kernel = [-1] * self.kernel_size + [0] + self.kernel_size * [1]
        conv_out = np.convolve(self.conv_col, kernel, "valid")
        conv_zeros_idx = []
        for idx, (conv_i, conv_j) in enumerate(zip(conv_out[:-1], conv_out[1:])):
            if conv_i * conv_j <= 0:
                if abs(conv_i) < abs(conv_j):
                    conv_zeros_idx.append(idx)
                else:
                    conv_zeros_idx.append(idx + 1)
        conv_zeros_idx = list(set(conv_zeros_idx))
        conv_zeros_idx.sort()
        zeros_idx = [idx + self.kernel_size for idx in conv_zeros_idx]
        self.correct_break_points(zeros_idx)

    def correct_break_points(self, zeros_idx):
        # classify peak_points into PEAK_POINT/POKE_POINT
        break_points_padded = [0] + zeros_idx + [len(self.df) - 1]
        start_idx = 0
        for i, (fp, sp, tp) in enumerate(
            zip(break_points_padded[:-2], break_points_padded[1:-1], break_points_padded[2:])
        ):
            fp = max(fp, start_idx)
            if self.conv_col.iloc[fp] > self.conv_col.iloc[sp] and self.conv_col.iloc[tp] > self.conv_col.iloc[sp]:
                idx_min = self.df[fp + 1 : tp][["Close", "Low"]].idxmin()
                start_idx = idx_min[0]
                self.add_poke_point(idx_min)
            elif self.conv_col.iloc[fp] < self.conv_col.iloc[sp] and self.conv_col.iloc[tp] < self.conv_col.iloc[sp]:
                idx_max = self.df[fp + 1 : tp][["Close", "High"]].idxmax()
                start_idx = idx_max[0]
                self.add_peak_point(idx_max)

    def fix_break_points(self):
        temp_zz_points = (
            [ZZPoint(0, POINT_TYPE.POKE_POINT, None)]
            + self.zz_points
            + [ZZPoint(len(self.df) - 1, POINT_TYPE.PEAK_POINT, None)]
        )
        for i, (fp, sp, tp) in enumerate(zip(temp_zz_points[:-2], temp_zz_points[1:-1], temp_zz_points[2:])):
            if sp.ptype == POINT_TYPE.PEAK_POINT:
                fixed_point = self.df[fp.pidx + 1 : tp.pidx][["Close", "High"]].idxmax()
                sp.pidx = fixed_point[0]
                sp.pline = SRLine(self.df.iloc[fixed_point[0]]["Close"], self.df.iloc[fixed_point[1]]["High"])
            else:
                fixed_point = self.df[fp.pidx + 1 : tp.pidx][["Close", "Low"]].idxmin()
                sp.pidx = fixed_point[0]
                sp.pline = SRLine(self.df.iloc[fixed_point[1]]["Low"], self.df.iloc[fixed_point[0]]["Close"])

    def add_poke_point(self, idx):
        self.zz_points.append(
            ZZPoint(idx[0], POINT_TYPE.POKE_POINT, SRLine(self.df.iloc[idx[1]]["Low"], self.df.iloc[idx[0]]["Close"]))
        )

    def add_peak_point(self, idx):
        self.zz_points.append(
            ZZPoint(idx[0], POINT_TYPE.PEAK_POINT, SRLine(self.df.iloc[idx[0]]["Close"], self.df.iloc[idx[1]]["High"]))
        )

    def merge_break_points(self, min_div):
        merge_break_points(self.zz_points, min_div)


def zigzag_conv(df, kernel_size, min_div):
    zz = ZigZag(df, kernel_size=kernel_size)
    zz.merge_break_points(min_div)
    return zz.zz_points


def join_zz_points(ftp, scp):
    # join 2 zz_points ftp, scp same type
    # return a joined zz_point
    if ftp.ptype == POINT_TYPE.PEAK_POINT:
        return ZZPoint(
            ftp.pidx if ftp.pline.low >= scp.pline.low else scp.pidx,
            ftp.ptype,
            SRLine(max(ftp.pline.low, scp.pline.low), max(ftp.pline.high, scp.pline.high)),
        )
    else:
        return ZZPoint(
            ftp.pidx if ftp.pline.high <= scp.pline.high else scp.pidx,
            ftp.ptype,
            SRLine(min(ftp.pline.low, scp.pline.low), min(ftp.pline.high, scp.pline.high)),
        )


def zigzag_conv_stream(df, kernel_size, min_div, zz_points):
    if len(zz_points) < 3:
        zz_points.clear()
        zz_points.extend(zigzag_conv(df, kernel_size, min_div))
        return
    idx = max(zz_points[-3].pidx - 5, 0)
    zzps = zigzag_conv(df[idx:].reset_index(), kernel_size, min_div)
    for zzp in zzps:
        zzp.pidx += idx
    zzps = [zp for zp in zzps if zp.pidx > zz_points[-1].pidx]
    if len(zzps) > 0:
        if zzps[0].ptype == zz_points[-1].ptype:
            ftp = zz_points.pop()
            scp = zzps[0]
            joined_zz_point = join_zz_points(ftp, scp)
            last_idx = len(zz_points) - 2
            zz_points.append(joined_zz_point)
            zz_points.extend(zzps[1:])
            merge_break_points(zz_points, min_div, from_idx=last_idx)
        else:
            last_idx = len(zz_points) - 2
            zz_points.extend(zzps)
            merge_break_points(zz_points, min_div, from_idx=last_idx)


def zigzag(df, sigma: float):
    up_zig = True  # Last extreme is a bottom. Next is a top.
    tmp_max = df.iloc[0]["High"]
    tmp_min = df.iloc[0]["Low"]
    tmp_max_i = 0
    tmp_min_i = 0
    close_max_i = 0
    close_min_i = 0
    zz_points = []
    for i in range(len(df)):
        if up_zig:  # Last extreme is a bottom
            if df.iloc[i]["High"] > tmp_max:
                # New high, update
                tmp_max = df.iloc[i]["High"]
                tmp_max_i = i
            elif (
                df.iloc[i]["Close"] < tmp_max - tmp_max * sigma
                and df.iloc[i]["Close"] < df.iloc[tmp_max_i]["Close"]
                and df.iloc[i]["Low"] < df.iloc[tmp_max_i]["Low"]
            ):
                # Price retraced by sigma %. Top confirmed, record it
                # zz_points[0] = type
                # zz_points[1] = index
                # zz_points[2] = price
                zz_points.append(
                    ZZPoint(close_max_i, POINT_TYPE.PEAK_POINT, SRLine(df.iloc[close_max_i]["Close"], tmp_max))
                )

                # Setup for next bottom
                up_zig = False
                tmp_min = df.iloc[i]["Low"]
                tmp_min_i = i
                close_min_i = i
            if df.iloc[i]["Close"] > df.iloc[close_max_i]["Close"]:
                close_max_i = i
        else:  # Last extreme is a top
            if df.iloc[i]["Low"] < tmp_min:
                # New low, update
                tmp_min = df.iloc[i]["Low"]
                tmp_min_i = i
            elif (
                df.iloc[i]["Close"] > tmp_min + tmp_min * sigma
                and df.iloc[i]["Close"] > df.iloc[tmp_min_i]["Close"]
                and df.iloc[i]["High"] > df.iloc[tmp_min_i]["High"]
            ):
                # Price retraced by sigma %. Bottom confirmed, record it
                # zz_points[0] = type
                # zz_points[1] = index
                # zz_points[2] = price
                zz_points.append(
                    ZZPoint(close_min_i, POINT_TYPE.POKE_POINT, SRLine(tmp_min, df.iloc[close_min_i]["Close"]))
                )

                # Setup for next top
                up_zig = True
                tmp_max = df.iloc[i]["High"]
                tmp_max_i = i
                close_max_i = i
            if df.iloc[i]["Close"] < df.iloc[close_min_i]["Close"]:
                close_min_i = i
    if len(zz_points) == 0:
        tmp_max_i = df["High"].idxmax()
        tmp_min_i = df["Low"].idxmin()
        if tmp_max_i < tmp_min_i:
            close_max_i = df["Close"].idxmax()
            zz_points.append(ZZPoint(close_max_i, POINT_TYPE.PEAK_POINT, SRLine(df.iloc[close_max_i]["Close"], df.iloc[tmp_max_i]["High"])))
        else:
            close_min_i = df["Close"].idxmin()
            zz_points.append(ZZPoint(close_min_i, POINT_TYPE.POKE_POINT, SRLine(df.iloc[tmp_min_i]["Low"], df.iloc[close_min_i]["Close"])))
    return zz_points


def zigzag_stream(df, sigma: float, zz_points):
    last_zz_points = zz_points[-1]
    i = len(df) - 1

    if last_zz_points.ptype == POINT_TYPE.POKE_POINT:  # Last extreme is a bottom
        tmp_max_i = df["High"].iloc[last_zz_points.pidx :].idxmax()
        close_max_i = df["Close"].iloc[last_zz_points.pidx :].idxmax()
        tmp_max = df["High"].iloc[tmp_max_i]
        if (
            df.iloc[i]["Close"] < tmp_max - tmp_max * sigma
            and df.iloc[i]["Close"] < df.iloc[tmp_max_i]["Close"]
            and df.iloc[i]["Low"] < df.iloc[tmp_max_i]["Low"]
        ):
            zz_points.append(
                ZZPoint(close_max_i, POINT_TYPE.PEAK_POINT, SRLine(df["Close"].iloc[close_max_i], tmp_max))
            )
    else:  # Last extreme is a top
        tmp_min_i = df["Low"].iloc[last_zz_points.pidx :].idxmin()
        close_min_i = df["Close"].iloc[last_zz_points.pidx :].idxmin()
        tmp_min = df["Low"].iloc[tmp_min_i]
        if (
            df.iloc[i]["Close"] > tmp_min + tmp_min * sigma
            and df.iloc[i]["Close"] > df.iloc[tmp_min_i]["Close"]
            and df.iloc[i]["High"] > df.iloc[tmp_min_i]["High"]
        ):
            zz_points.append(
                ZZPoint(close_min_i, POINT_TYPE.POKE_POINT, SRLine(tmp_min, df["Close"].iloc[close_min_i].min()))
            )
