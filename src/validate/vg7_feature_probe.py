# -*- coding: utf-8 -*-
"""VG-7 特徵真偽篩（v2.14，2026/7/15——vol_probe 三檢定收編為標準關卡）。

任何在診斷 A/C 通過 Holm 的候選特徵，宣告「含訊號」之前必經本關卡，
排除「二元標籤 × 固定成本門檻 × 離散度」機械假象（首例：realized_vol_20d，
離散度 r=0.295 vs 均值 r=0.030，H_artifact 定案）。

統計實作採已驗證版本（提案原始碼三處問題之修正，L23）：
    P1 市場層級占比：groupby 日均 + ICC 校正 (R²−1/k)/(1−1/k)
       （日期虛擬變數 OLS 之 R² 數學等價且同帶 1/k 偏誤，並需 statsmodels）
    P2 擇時：日均特徵 vs 基準未來 N 日淨報酬，每隔 N 日取樣 Spearman
       （逐日全取樣之 N 日報酬互相重疊 → 時序偽重複）
    P3 均值/離散度效應：獨立子樣本 Spearman（全列 pearson = 橫斷面偽重複，
       L13/L20 同款錯誤第三次出現，於此制度化封死）
    三檢定 p 值過 Holm(m=3)。

結局矩陣（八格窮舉，未落入乾淨格 → 需人工複核，不自動套布林）：
    均值✓（不論其餘）        → 特徵對報酬均值含資訊 → 進折內回歸+IC
    均值✗ 離散✓ 擇時✓        → 二元關聯屬假象，但擇時訊號真實 → 擇時層立項
    均值✗ 離散✓ 擇時✗        → H_artifact：機械假象，不可交易，歸檔
    均值✗ 離散✗ 擇時✓        → 未預期格：純時序訊號？需人工複核
    均值✗ 離散✗ 擇時✗        → 證據不足，歸檔待前瞻資料
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.validate.vg3_significance import holm_correction


@dataclass(frozen=True)
class VG7Report:
    feature: str
    market_level_share_adj: float      # P1（ICC 校正，僅描述不進 Holm）
    timing_ic: float | None
    timing_p_holm: float
    mean_r: float
    mean_p_holm: float
    dispersion_r: float
    dispersion_p_holm: float
    n_independent: int
    verdict: str
    is_clean_cell: bool                # False = 需人工複核


def market_level_share(dev: pd.DataFrame, col: str) -> float:
    """P1：ICC 校正之市場層級占比（純雜訊 → ≈0；細節見模組 docstring）。"""
    x = dev[[col, "date"]].dropna()
    if len(x) < 10:
        return 0.0
    daily_mean = x.groupby("date")[col].transform("mean")
    resid = x[col] - daily_mean
    tot = float(x[col].var(ddof=1))
    if tot <= 0:
        return 0.0
    raw = 1.0 - float(resid.var(ddof=1)) / tot
    k_mean = float(x.groupby("date")[col].size().mean())
    baseline = 1.0 / k_mean if k_mean > 1 else 0.0
    return round(max(0.0, (raw - baseline) / (1.0 - baseline)), 4) \
        if baseline < 1 else 0.0


def timing_ic(dev: pd.DataFrame, bench: pd.DataFrame, p1_cfg, n_days: int,
              col: str) -> dict:
    """P2：日均特徵 vs 基準未來 N 日淨報酬（每隔 N 日獨立取樣）。"""
    daily = dev.dropna(subset=[col]).groupby("date")[col].mean()
    b = bench.sort_values("date").reset_index(drop=True)
    cost = p1_cfg.fee_buy_rate + p1_cfg.fee_sell_rate + p1_cfg.tax_sell_rate
    fwd = pd.Series(
        (b["close"].shift(-n_days) / b["open"].shift(-1) - 1 - cost).to_numpy(),
        index=pd.to_datetime(b["date"]).dt.normalize())
    idx = daily.index.intersection(fwd.index)[::n_days]
    x, y = daily.reindex(idx), fwd.reindex(idx)
    m = x.notna() & y.notna()
    if m.sum() < 30:
        return {"ic": None, "p": 1.0, "n": int(m.sum())}
    ic, p = spearmanr(x[m], y[m])
    return {"ic": round(float(ic), 4), "p": float(p), "n": int(m.sum())}


def mean_dispersion_effects(sub: pd.DataFrame, col: str) -> dict:
    """P3：獨立子樣本上，特徵對報酬「均值」與「離散度」的 Spearman 分離。"""
    d = sub.dropna(subset=[col, "fwd_return_net"])
    x = d[col].astype(float)
    r_m, p_m = spearmanr(x, d["fwd_return_net"].astype(float))
    r_d, p_d = spearmanr(x, d["fwd_return_net"].abs().astype(float))
    return {"r_mean": round(float(r_m), 4), "p_mean": float(p_m),
            "r_disp": round(float(r_d), 4), "p_disp": float(p_d),
            "n": int(len(d))}


def probe_feature_artifact(
    dev: pd.DataFrame, sub: pd.DataFrame, bench: pd.DataFrame,
    feature_col: str, p1_cfg, p2_cfg,
) -> VG7Report:
    """VG-7 主函式：三檢定 + Holm(3) + 八格窮舉判定。"""
    share = market_level_share(dev, feature_col)
    t = timing_ic(dev, bench, p1_cfg, p2_cfg.forward_return_days, feature_col)
    e = mean_dispersion_effects(sub, feature_col)
    _, adj = holm_correction([t["p"], e["p_mean"], e["p_disp"]])
    t_sig, m_sig, d_sig = (x < 0.05 for x in adj)

    clean = True
    if m_sig:
        verdict = "均值含資訊 → 進 walk-forward 折內回歸+IC 評估"
    elif d_sig and t_sig:
        verdict = "二元關聯屬離散度假象，但擇時訊號真實 → 擇時層立項；選股層無訊號"
    elif d_sig:
        verdict = "H_artifact：二元標籤×固定成本門檻×離散度機械假象，不可交易，歸檔"
    elif t_sig:
        verdict = "未預期格（僅擇時顯著、均值/離散皆✗）→ 需人工複核，不自動判定"
        clean = False
    else:
        verdict = "三檢定皆不顯著 → 證據不足，歸檔待前瞻資料"
    return VG7Report(
        feature=feature_col, market_level_share_adj=share,
        timing_ic=t["ic"], timing_p_holm=round(adj[0], 4),
        mean_r=e["r_mean"], mean_p_holm=round(adj[1], 4),
        dispersion_r=e["r_disp"], dispersion_p_holm=round(adj[2], 4),
        n_independent=e["n"], verdict=verdict, is_clean_cell=clean,
    )
