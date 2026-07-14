# -*- coding: utf-8 -*-
"""VG-4 樣本量與有效性：沿用定案4統計獨立層計數（VG-3/4 唯一依據）。"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from config.phase2_config import Phase2Config
from src.signal_events import (canonical_independent_samples,
                               extract_signal_events)


@dataclass(frozen=True)
class VG4Report:
    n_events: int
    n_independent: int
    oos_period_days: int
    oos_n_independent: int
    threshold: int
    reliable: bool
    statement: str


def build_vg4_report(signal: pd.Series, oos_signal: pd.Series,
                     cfg: Phase2Config) -> VG4Report:
    # v2.6：一律走正典管線（逐日聚合→事件→≥N），修復多股 concat 低估
    daily = signal.groupby(signal.index).any().sort_index()
    ev = extract_signal_events(daily)
    n_ind, _, _ = canonical_independent_samples(
        signal, None, cfg.forward_return_days)
    oos_ind, _, _ = canonical_independent_samples(
        oos_signal, None, cfg.forward_return_days)
    oos_days = (int((oos_signal.index.max() - oos_signal.index.min()).days)
                if len(oos_signal) else 0)
    reliable = n_ind >= cfg.min_independent_signals
    stmt = (f"獨立訊號數 {n_ind}（事件數 {len(ev)}），門檻 {cfg.min_independent_signals}。"
            + ("達標。" if reliable else
               "【樣本數過少，統計結果不可靠】不可據以採信高勝率。")
            + f" 樣本外期間 {oos_days} 天、獨立訊號 {oos_ind} 個"
            + ("（樣本外訊號過少，需一併警示）。" if oos_ind < 5 else "。"))
    return VG4Report(len(ev), n_ind, oos_days, oos_ind,
                     cfg.min_independent_signals, reliable, stmt)
