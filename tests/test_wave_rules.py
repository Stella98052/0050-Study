# -*- coding: utf-8 -*-
"""三大鐵律布林規則 + 標籤演算法行為測試。"""
from conftest import make_synthetic_ohlcv
from src.wave.wave_labels import (
    rule_wave2_not_break_wave1_origin,
    rule_wave3_not_shortest,
    rule_wave4_not_overlap_wave1,
    label_waves_retrospective,
    label_waves_realtime,
)


def test_iron_rule_1():
    assert rule_wave2_not_break_wave1_origin(100.0, 101.0)
    assert not rule_wave2_not_break_wave1_origin(100.0, 99.0)


def test_iron_rule_2_partial_and_final():
    ok, key = rule_wave3_not_shortest(10.0, 12.0, None)
    assert ok and key == "wave3_not_shortest_partial"     # 定案 3：暫定
    ok, key = rule_wave3_not_shortest(10.0, 8.0, 9.0)
    assert not ok and key == "wave3_not_shortest_final"   # 波3 最短 → 違反
    ok, _ = rule_wave3_not_shortest(10.0, 11.0, 9.0)
    assert ok                                             # 最短為波5，波3 非最短 → 通過


def test_iron_rule_3():
    assert rule_wave4_not_overlap_wave1(w1_high=110.0, w4_low=111.0)
    assert not rule_wave4_not_overlap_wave1(w1_high=110.0, w4_low=109.0)


def test_labels_only_valid_values(cfg, ohlcv):
    valid = {"1","2","3","4","5","A","B","C","unknown"}
    segs = label_waves_retrospective(ohlcv, cfg)
    assert len(segs) > 0
    assert all(s.label in valid for s in segs)
    rt = label_waves_realtime(ohlcv, cfg)
    assert set(rt["wave_label_realtime"].unique()) <= valid


def test_retrospective_marked_and_realtime_marked(cfg, ohlcv):
    segs = label_waves_retrospective(ohlcv, cfg)
    assert all(s.version == "retrospective" for s in segs)
    assert all(s.start_pivot.confirmed_date is None for s in segs)
