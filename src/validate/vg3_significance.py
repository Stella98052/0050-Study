# -*- coding: utf-8 -*-
"""VG-3 統計顯著性：permutation test（預設）與 bootstrap 信賴區間。
不可僅呈現單一績效數字而不附顯著性資訊。

【輸入要求（v2.4，定案4原文）】signal_returns 必須是
signal_events.independent_return_series 篩出的統計獨立子集；
餵入未篩選的重疊交易層序列 = 偽重複，會人為壓低 p 值 / 收窄 CI（L13）。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.phase2_config import Phase2Config


@dataclass(frozen=True)
class VG3Report:
    method: Literal["permutation", "bootstrap"]
    n_iterations: int
    p_value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    metric_name: str
    plain_language: str
    passed: bool


def permutation_test(signal_returns: pd.Series, all_returns: pd.Series,
                     cfg: Phase2Config) -> VG3Report:
    """打亂訊號標籤重跑 N 次（自全體報酬池隨機抽同樣本數），
    p_value = 隨機平均報酬 ≥ 真實平均報酬 的比例。"""
    sig = signal_returns.dropna().to_numpy()
    pool = all_returns.dropna().to_numpy()
    if len(sig) == 0 or len(pool) < len(sig):
        return VG3Report("permutation", 0, None, None, None, "mean_return",
                         "樣本不足，無法檢定。", False)
    rng = np.random.default_rng(cfg.random_state)
    real = sig.mean()
    perm = np.array([rng.choice(pool, size=len(sig), replace=False).mean()
                     for _ in range(cfg.permutation_n)])
    p = float((perm >= real).mean())
    passed = p < (1.0 - cfg.confidence_level)
    plain = (f"真實訊號平均報酬 {real:.4%}，落在 {cfg.permutation_n} 次隨機抽樣"
             f"分布的第 {(1-p)*100:.1f} 百分位（p={p:.4f}）。"
             + ("→ 顯著優於隨機。" if passed else
                "→ 無法證明此訊號優於隨機（p 未達顯著水準）。"))
    return VG3Report("permutation", cfg.permutation_n, round(p, 4),
                     None, None, "mean_return", plain, passed)


def bootstrap_ci(signal_returns: pd.Series,
                 metric: Literal["sharpe", "mean_return"],
                 cfg: Phase2Config, holding_days: int = 5) -> VG3Report:
    """重抽樣 N 次 → 指標 95% CI；CI 含 0 → 無法證明優於隨機。"""
    from src.model.metrics import _sharpe
    sig = signal_returns.dropna()
    if len(sig) < 5:
        return VG3Report("bootstrap", 0, None, None, None, metric,
                         "樣本不足，無法建立信賴區間。", False)
    rng = np.random.default_rng(cfg.random_state)
    vals = []
    arr = sig.to_numpy()
    for _ in range(cfg.bootstrap_n):
        s = pd.Series(rng.choice(arr, size=len(arr), replace=True))
        vals.append(_sharpe(s, holding_days) if metric == "sharpe"
                    else float(s.mean()))
    lo, hi = np.percentile(vals, [(1-cfg.confidence_level)/2*100,
                                  (1+cfg.confidence_level)/2*100])
    passed = lo > 0
    plain = (f"{metric} 之 {cfg.confidence_level:.0%} 信賴區間 "
             f"[{lo:.4f}, {hi:.4f}]。"
             + ("→ 區間不含 0，顯著為正。" if passed else
                "→ 信賴區間包含 0，代表無法證明此訊號優於隨機。"))
    return VG3Report("bootstrap", cfg.bootstrap_n, None,
                     round(float(lo), 4), round(float(hi), 4),
                     metric, plain, passed)


def holm_correction(p_values: list[float], alpha: float = 0.05
                    ) -> tuple[list[bool], list[float]]:
    """Holm 逐步多重比較校正（B1，v2.7）。回傳 (逐項是否顯著, 調整後p)。

    m 次檢定中，純雜訊下至少一次假顯著機率 = 1−(1−α)^m（m=5 → 22.6%），
    故 N 網格逐點顯著性必須經校正後才可宣稱。
    調整後 p_i = max_{j≤i(排序)} min(1, (m−rank_j)×p_j)（單調化）。"""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i] is None, p_values[i]))
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        p = p_values[i]
        if p is None:
            adj[i] = 1.0
            continue
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[i] = round(running, 6)
    reject = [a < alpha for a in adj]
    return reject, adj
