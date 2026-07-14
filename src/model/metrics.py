# -*- coding: utf-8 -*-
"""績效指標 v2.3（gross/net 並列；全部公式明列）。★

【版本演進（教訓鏈，詳見 LESSONLEARNT L9/L10）】
v2.0 逐筆連乘 → 虛構重疊複利（holdout 曾算出 1,677萬%）
v2.1 等權日籃連乘 → 量級仍被時間壓縮 N 倍
v2.2 日等效 (1+r)^(1/N) 鏈乘 → 指數律恆等於 (∏(1+r))^(1/N)，
     不對應任何可執行路徑；僅在退化（稠密均勻）案例巧合逼近正確值，
     且當時的鎖定測試恰用退化案例自我印證（外部審查 2026/7/12 指出並經驗算確認）
v2.3（現版，採納外部審查）：
     權益曲線 = 「不重疊交易」原始報酬直接連乘。
     不重疊（同 VG-4 規則：間隔 ≥ N 日曆日，共用 select_independent_dates）
     的交易對應真實可執行的單部位策略——上一筆平倉才進下一筆——
     連乘即真實路徑，無需任何換算。
     同日多筆交易先等權合併為一筆（單部位、等權進場近似 approximation）。

指標定義：
    交易層（全部交易，不受重疊影響）：n_trades / 勝率 / 盈虧比 / avg_trade_net
    路徑層（僅不重疊交易）：total_return / MDD / Sharpe / Alpha
    Sharpe = mean(獨立交易報酬)/std × √(252/持有天數)
    MDD = 沿獨立交易淨值曲線 max((峰−谷)/峰)
    Alpha = 獨立路徑總報酬 − 同期 0050 買進持有
    n_independent < VG-4 門檻時，路徑層數字不具統計意義，
    呼叫端必須以 VG-4 警語蓋過，不得單獨呈現。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.signal_events import independent_return_series


@dataclass(frozen=True)
class BacktestMetrics:
    n_trades: int
    n_independent: int                 # 不重疊交易數（=路徑層樣本數，對齊VG-4）
    avg_trade_net: float               # 單筆平均淨報酬（交易層，全部交易）
    sharpe_gross: float
    sharpe_net: float
    mdd_net: float
    win_rate_net: float
    payoff_ratio_net: float
    alpha_net_vs_benchmark: float
    total_return_gross: float
    total_return_net: float


@dataclass(frozen=True)
class FoldResult:
    split: object
    metrics: BacktestMetrics
    class_balance: dict
    feature_importance: dict


def _sharpe(r: pd.Series, holding_days: int) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252.0 / holding_days))


def _independent_series(returns: pd.Series, dates, n_days: int) -> pd.Series:
    """委派 signal_events.independent_return_series（單一事實來源，L12）。"""
    return independent_return_series(returns, dates, n_days)


def _equity_stats(indep: pd.Series) -> tuple[float, float]:
    """(總報酬, MDD)：不重疊交易原始報酬直接連乘（真實可執行路徑）。"""
    if len(indep) == 0:
        return 0.0, 0.0
    equity = (1.0 + indep).cumprod()
    peak = equity.cummax()
    return float(equity.iloc[-1] - 1.0), float(((peak - equity) / peak).max())


def compute_backtest_metrics(
    gross: pd.Series, net: pd.Series, dates,
    benchmark_return: float, holding_days: int,
) -> BacktestMetrics:
    g, n = gross.dropna(), net.dropna()
    wins, losses = n[n > 0], n[n <= 0]
    payoff = (float(wins.mean() / abs(losses.mean()))
              if len(wins) and len(losses) and losses.mean() != 0 else 0.0)
    ig = _independent_series(gross, dates, holding_days)
    inn = _independent_series(net, dates, holding_days)
    tot_g, _ = _equity_stats(ig)
    tot_n, mdd_n = _equity_stats(inn)
    return BacktestMetrics(
        n_trades=int(len(n)),
        n_independent=int(len(inn)),
        avg_trade_net=round(float(n.mean()) if len(n) else 0.0, 6),
        sharpe_gross=round(_sharpe(ig, holding_days), 4),
        sharpe_net=round(_sharpe(inn, holding_days), 4),
        mdd_net=round(mdd_n, 4),
        win_rate_net=round(float((n > 0).mean()) if len(n) else 0.0, 4),
        payoff_ratio_net=round(payoff, 4),
        alpha_net_vs_benchmark=round(tot_n - benchmark_return, 4),
        total_return_gross=round(tot_g, 4),
        total_return_net=round(tot_n, 4),
    )
