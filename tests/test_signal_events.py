# -*- coding: utf-8 -*-
"""定案 4 雙軌計數測試 + 13MV 絕對否決納入訊號測試。"""
import pandas as pd

from config.config import Config
from src.signal_events import (
    extract_signal_events,
    count_statistically_independent_signals,
    build_signal_count_report,
)


def _sig(dates_true, all_dates):
    return pd.Series(
        [d in dates_true for d in all_dates],
        index=pd.DatetimeIndex(all_dates),
    )


def test_event_layer_merges_consecutive_days(cfg):
    days = pd.bdate_range("2025-01-01", periods=20)
    true_days = set(days[2:5]) | set(days[8:9]) | set(days[12:15])   # 3 事件
    events = extract_signal_events(_sig(true_days, days))
    assert len(events) == 3
    assert events[0] == days[2] and events[1] == days[8] and events[2] == days[12]


def test_independent_layer_merges_overlapping_windows(cfg):
    """事件間隔 < N 天 → 報酬窗口重疊 → 合併為同一統計樣本。"""
    days = pd.bdate_range("2025-01-01", periods=40)
    # 事件起點：day0, day+2(重疊), day+10(獨立), day+11(重疊)
    starts = [days[0], days[2], days[10], days[11]]
    n, kept = count_statistically_independent_signals(starts, n_days=5)
    assert n == 2
    assert kept == [days[0], days[10]]


def test_dual_track_report(cfg):
    days = pd.bdate_range("2025-01-01", periods=30)
    # 事件起點 days[0]、days[2]（相隔 2 日曆日 < 5 → 窗口重疊合併）、days[15]
    true_days = set(days[0:1]) | set(days[2:3]) | set(days[15:16])
    rpt = build_signal_count_report(_sig(true_days, days), cfg)
    assert rpt.n_events == 3
    assert rpt.n_independent == 2          # 前兩事件窗口重疊（N=5）
    assert "統計獨立樣本數" in rpt.note and "重疊" in rpt.note


def test_mv13_veto_blocks_signal(cfg):
    """13MV 下彎日即使其餘條件全滿足，訊號必為 False（絕對否決）。"""
    from src.signal_events import detect_wave3_tidal_burst
    days = pd.bdate_range("2025-01-01", periods=3)
    wl = pd.DataFrame({"date": days, "wave_label_realtime": ["3", "3", "3"]})
    mv = pd.DataFrame({
        "date": days,
        "is_volume_burst": [True, True, True],
        "mv_short_direction": [1, 1, 1],
        "mv_mid_veto_active": [False, True, False],     # 中日 13MV 下彎
    })
    div = pd.Series([False, False, False])
    sig = detect_wave3_tidal_burst(wl, mv, div, cfg)
    assert sig.tolist() == [True, False, True]
