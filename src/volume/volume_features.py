# -*- coding: utf-8 -*-
"""MV 潮汐量能特徵（定案 2：5 / 13 / 20 三線，13MV 為方法論核心否決線）。

全部欄位僅使用截至當日之歷史（pandas rolling 天然無未來函數）。

輸出欄位：
    mv_short / mv_mid / mv_long        5MV / 13MV / 20MV（成交量均線）
    mv_short_direction / mv_mid_direction / mv_long_direction  +1↑ / -1↓ / 0持平
    mv_bias                            量能乖離率 = mv_short / mv_long - 1
    is_volume_burst                    mv_bias > volume_burst_bias_threshold
    mv_mid_is_core_veto_line           恆為 True 之標記欄（定案 2 要求：
                                       13MV 明確標記為核心否決線，供第二階段
                                       模型與第三階段面板優先呈現）
    mv_mid_veto_active                 13MV 當日下彎（方法論：絕對否決）
"""

from __future__ import annotations

import pandas as pd

from config.config import Config


def _direction(series: pd.Series) -> pd.Series:
    """均線方向：與前一日比較，上升 +1、下降 -1、持平/不足資料 0。"""
    diff = series.diff()
    return diff.apply(
        lambda x: 0 if pd.isna(x) or x == 0 else (1 if x > 0 else -1)
    ).astype("int64")


def compute_mv_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """★ 計算三線 MV 特徵（可安全用於模型訓練）。

    calc_logic：mv_k = volume.rolling(k).mean()；方向 = 當日均值 − 前日均值之
    符號；乖離率 = mv_short / mv_long − 1；窗口未滿之日輸出 NaN / 方向 0。
    """
    vol = df["volume"].astype("float64")
    mv_s = vol.rolling(cfg.vol_ma_short).mean()
    mv_m = vol.rolling(cfg.vol_ma_mid).mean()
    mv_l = vol.rolling(cfg.vol_ma_long).mean()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.normalize(),
            "mv_short": mv_s,
            "mv_mid": mv_m,
            "mv_long": mv_l,
            "mv_short_direction": _direction(mv_s),
            "mv_mid_direction": _direction(mv_m),
            "mv_long_direction": _direction(mv_l),
        }
    )
    out["mv_bias"] = out["mv_short"] / out["mv_long"] - 1.0
    out["is_volume_burst"] = (
        out["mv_bias"] > cfg.volume_burst_bias_threshold
    ).fillna(False)
    out["mv_mid_is_core_veto_line"] = True                  # 定案 2 標記
    out["mv_mid_veto_active"] = out["mv_mid_direction"] < 0  # 13MV 下彎 = 絕對否決
    return out


def detect_price_volume_divergence(
    df: pd.DataFrame, mv_features: pd.DataFrame, wave_labels_rt: pd.DataFrame
) -> pd.Series:
    """★ 量價背離（防未來函數版本）。

    定義（規格）：realtime 標籤為第 3 浪期間，價格創波段新高、
    但 5MV 未同步走強（mv_short_direction <= 0）→ True。

    「創新高」比較基準：僅使用「前一日以前」之歷史最高收盤
    （expanding().max().shift(1)），不含當日，防未來函數。
    """
    close = df["close"].astype("float64")
    prior_high = close.expanding().max().shift(1)
    new_high = close > prior_high

    is_wave3 = (wave_labels_rt["wave_label_realtime"] == "3").to_numpy()
    weak_vol = (mv_features["mv_short_direction"] <= 0).to_numpy()

    div = pd.Series(
        new_high.to_numpy() & is_wave3 & weak_vol,
        index=df.index,
        name="price_volume_divergence",
    )
    return div.fillna(False)
