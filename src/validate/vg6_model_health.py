# -*- coding: utf-8 -*-
"""VG-6 模型輸出健康度（v2.9，2026/7/14 審查建議升格為正式關卡）。

檢查對象：holdout 預測機率分布與判別力——「模型是否對輸入真的敏感、
還是退化成常數/系統性偏多輸出」，此前五道關卡皆未覆蓋。

檢查項（任一觸發 → passed=False）：
    near_constant_output   機率標準差 < 0.02（近常數輸出）
    no_discrimination_band 機率落在 [0.45, 0.55] 的占比 > 90%（無判別帶）
    one_sided_output       逐列選中率 > 95% 或 < 5%（系統性單邊）
    auc_no_skill           對 holdout 真實標籤之 AUC < 0.55（無判別力）

重要澄清（防誤判，L19）：多股池「任一檔/日」聚合占比 100% 是數學必然
（逐列選中率 41.5% × 10 檔 → 任一日 ≈99.5%），不得單獨作為退化證據；
本關卡以「逐列」機率分布與 AUC 為準。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VG6Report:
    n: int
    proba_min: float
    proba_p25: float
    proba_p50: float
    proba_p75: float
    proba_max: float
    proba_std: float
    pick_rate_row: float          # 逐列 proba>0.5 比例（非多股聚合占比）
    share_mid_band: float         # 機率落在 [0.45,0.55] 的占比
    auc: float | None             # 對真實標籤之判別力；標籤不可用時 None
    flags: tuple[str, ...]
    passed: bool
    statement: str


def manual_auc(proba: np.ndarray, labels: np.ndarray) -> float | None:
    """免 sklearn 之 AUC（Mann–Whitney U / 平均秩法，含平手處理）。

    AUC = (正類秩和 − n_pos(n_pos+1)/2) / (n_pos × n_neg)。"""
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(proba).rank(method="average").to_numpy()
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def vg6_model_output_health(
    proba, labels=None,
    std_floor: float = 0.02, mid_band_cap: float = 0.90,
    one_sided_hi: float = 0.95, auc_floor: float = 0.55,
) -> VG6Report:
    """執行 VG-6 全部檢查並回傳結構化報告（門檻為參數，預設值如上）。"""
    p = np.asarray(proba, dtype=float)
    q = np.percentile(p, [25, 50, 75])
    std = float(p.std(ddof=1)) if len(p) > 1 else 0.0
    pick = float((p > 0.5).mean())
    mid = float(((p >= 0.45) & (p <= 0.55)).mean())
    auc = None
    if labels is not None:
        lab = pd.Series(labels).astype(bool).to_numpy()
        auc = manual_auc(p, lab)

    flags: list[str] = []
    if std < std_floor:
        flags.append("near_constant_output")
    if mid > mid_band_cap:
        flags.append("no_discrimination_band")
    if pick > one_sided_hi or pick < (1 - one_sided_hi):
        flags.append("one_sided_output")
    if auc is not None and auc < auc_floor:
        flags.append("auc_no_skill")

    passed = not flags
    stmt = (f"機率分布 min={p.min():.3f} P25={q[0]:.3f} 中位={q[1]:.3f} "
            f"P75={q[2]:.3f} max={p.max():.3f} std={std:.3f}｜"
            f"逐列選中率={pick:.1%}｜中間帶占比={mid:.1%}"
            + (f"｜AUC={auc:.3f}" if auc is not None else "｜AUC=不可算")
            + ("。通過：模型輸出具分散度與判別力。" if passed else
               f"。未通過：{flags}——模型輸出退化，"
               "holdout 績效不可歸因於模型判斷，需重新設計特徵/訓練。")
            + "（註：多股池聚合占比100%為數學必然，本關卡以逐列分布為準）")
    return VG6Report(
        n=int(len(p)), proba_min=round(float(p.min()), 4),
        proba_p25=round(float(q[0]), 4), proba_p50=round(float(q[1]), 4),
        proba_p75=round(float(q[2]), 4), proba_max=round(float(p.max()), 4),
        proba_std=round(std, 4), pick_rate_row=round(pick, 4),
        share_mid_band=round(mid, 4),
        auc=(round(auc, 4) if auc is not None else None),
        flags=tuple(flags), passed=passed, statement=stmt,
    )
