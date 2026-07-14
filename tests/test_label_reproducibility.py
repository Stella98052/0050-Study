# -*- coding: utf-8 -*-
"""標籤重現性：固定種子，同一段資料重跑兩次結果逐一相等。"""
from conftest import make_synthetic_ohlcv
from src.wave.wave_labels import label_waves_retrospective, label_waves_realtime


def test_retrospective_reproducible(cfg):
    df1 = make_synthetic_ohlcv(seed=42)
    df2 = make_synthetic_ohlcv(seed=42)
    s1 = label_waves_retrospective(df1, cfg)
    s2 = label_waves_retrospective(df2, cfg)
    assert [(x.label, x.start_pivot, x.end_pivot) for x in s1] == \
           [(x.label, x.start_pivot, x.end_pivot) for x in s2]


def test_realtime_reproducible(cfg):
    df = make_synthetic_ohlcv(seed=42)
    r1 = label_waves_realtime(df, cfg)
    r2 = label_waves_realtime(df.copy(), cfg)
    assert r1["wave_label_realtime"].tolist() == r2["wave_label_realtime"].tolist()
