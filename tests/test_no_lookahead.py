# -*- coding: utf-8 -*-
"""VG-5 前身：realtime 輸出防未來函數（防前視偏誤）測試。"""
import pandas as pd

from conftest import make_synthetic_ohlcv
from src.wave.zigzag import compute_pivots_realtime
from src.wave.wave_labels import label_waves_realtime


def test_realtime_pivots_ignore_future_data(cfg):
    """竄改 as_of 之後的資料，realtime 轉折輸出必須完全不變。"""
    df = make_synthetic_ohlcv()
    as_of = df["date"].iloc[150].date()

    base = compute_pivots_realtime(df, as_of, cfg)

    tampered = df.copy()
    mask = tampered["date"].dt.date > as_of
    tampered.loc[mask, ["open", "high", "low", "close"]] = 999.0   # 極端竄改
    tampered.loc[mask, "volume"] = 1

    after = compute_pivots_realtime(tampered, as_of, cfg)
    assert base == after, "未來資料異動影響了 realtime 轉折輸出（前視洩漏）"


def test_realtime_pivot_confirmed_not_after_asof(cfg):
    df = make_synthetic_ohlcv()
    as_of = df["date"].iloc[180].date()
    for p in compute_pivots_realtime(df, as_of, cfg):
        assert p.confirmed_date is not None
        assert p.confirmed_date <= as_of
        assert p.pivot_date <= p.confirmed_date, "確認日不可早於轉折日"


def test_realtime_label_basis_not_after_date(cfg, ohlcv):
    """label_waves_realtime 逐列斷言：標籤依據之最後確認轉折日 <= 當日。"""
    labels = label_waves_realtime(ohlcv, cfg)
    valid = labels.dropna(subset=["label_basis_last_confirmed_date"])
    assert len(valid) > 0, "整段資料皆無確認轉折，測試資料不足"
    assert (
        valid["label_basis_last_confirmed_date"] <= valid["date"]
    ).all(), "存在標籤依據晚於當日之列（前視洩漏）"


def test_realtime_labels_ignore_future_data(cfg):
    """竄改尾段資料，前段每日 realtime 標籤必須逐列一致。"""
    df = make_synthetic_ohlcv()
    cut = 150
    as_of_labels = label_waves_realtime(df, cfg).iloc[:cut]

    tampered = df.copy()
    tampered.loc[cut:, ["open", "high", "low", "close"]] = 999.0
    tampered_labels = label_waves_realtime(tampered, cfg).iloc[:cut]

    pd.testing.assert_series_equal(
        as_of_labels["wave_label_realtime"],
        tampered_labels["wave_label_realtime"],
    )
