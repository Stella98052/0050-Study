# -*- coding: utf-8 -*-
"""MV 潮汐量能特徵測試（定案 2：三線 + 13MV 核心否決）。"""
import numpy as np
import pandas as pd

from conftest import make_synthetic_ohlcv
from src.volume.volume_features import compute_mv_features


def test_three_lines_and_veto_flag(cfg, ohlcv):
    f = compute_mv_features(ohlcv, cfg)
    for col in ["mv_short", "mv_mid", "mv_long",
                "mv_mid_is_core_veto_line", "mv_mid_veto_active"]:
        assert col in f.columns
    assert f["mv_mid_is_core_veto_line"].all()          # 定案 2 標記恆為 True


def test_rolling_matches_manual(cfg, ohlcv):
    """calc_logic 驗算：第 30 列 5MV = 第 26~30 列成交量手工平均。"""
    f = compute_mv_features(ohlcv, cfg)
    manual = ohlcv["volume"].iloc[26:31].mean()
    assert abs(f["mv_short"].iloc[30] - manual) < 1e-6


def test_veto_active_when_mv13_declines(cfg):
    """量能持續遞減 → 13MV 必下彎 → veto_active=True。"""
    n = 60
    df = make_synthetic_ohlcv(n=n)
    df["volume"] = np.linspace(20_000_000, 5_000_000, n).astype("int64")
    f = compute_mv_features(df, cfg)
    tail = f.iloc[cfg.vol_ma_mid + 2:]
    assert tail["mv_mid_veto_active"].all()


def test_burst_flag_threshold(cfg):
    """量能突然放大 → mv_bias 突破門檻 → is_volume_burst=True。"""
    n = 60
    df = make_synthetic_ohlcv(n=n)
    df["volume"] = 10_000_000
    df.loc[n - 5:, "volume"] = 40_000_000              # 尾端爆量
    f = compute_mv_features(df, cfg)
    assert f["is_volume_burst"].iloc[-1]
    assert not f["is_volume_burst"].iloc[30]
