# -*- coding: utf-8 -*-
"""評分模型參數（v3.27）——權重與標準化門檻集中於此，禁止散落硬編碼。

權重採**聖杯（費波那契/黃金比例）遞減**：13:8:5:3:2:1，相鄰比值趨近
φ=1.618。順序即重要性排序，改順序或改數列都只需改本檔一行。

門檻為初始校準值（engineering-default），後續依實例調校；每次調整
須記錄理由與日期，避免無憑據的參數漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 聖杯比例權重：相鄰項比值 ≈ 0.618（黃金比例）
FIB_WEIGHTS = [13, 8, 5, 3, 2, 1]

# 因子順序＝重要性排序（依「基本面獲利成長 → 籌碼動能 → 技術確認」）
FACTOR_ORDER = ["growth", "quality", "institution", "margin", "tide",
                "valuation"]

FACTOR_LABEL = {
    "growth": "成長動能（營收YoY＋EPS加速度）",
    "quality": "獲利品質（三率改善＋ROE＋FCF）",
    "institution": "法人動能（買超強度＋連買天數）",
    "margin": "籌碼結構（融資退場＋券資比）",
    "tide": "量價動能（13MV 潮汐）",
    "valuation": "估值位階（本益比百分位）",
}


def golden_weights() -> dict[str, float]:
    """費波那契權重正規化為百分比（總和 100）。"""
    total = sum(FIB_WEIGHTS)
    return {k: round(w / total * 100.0, 2)
            for k, w in zip(FACTOR_ORDER, FIB_WEIGHTS)}


@dataclass(frozen=True)
class ScoreConfig:
    """標準化門檻（達此值即該分項滿分；皆為 engineering-default 初值）。"""

    weights: dict = field(default_factory=golden_weights)

    # 成長動能
    rev_yoy_full: float = 0.40          # 月營收 YoY 40% → 滿分
    eps_accel_full: float = 0.30        # EPS 年增加速度 30 百分點 → 滿分
    # 獲利品質
    margin_improve_full: float = 0.03   # 毛利率年增 3 個百分點 → 滿分
    roe_full: float = 0.20              # ROE 20%（年化）→ 滿分
    roe_floor: float = 0.05             # ROE 5% 以下 → 0 分
    # 法人動能
    inst_ratio_full: float = 0.10       # 淨買超佔成交量 10% → 滿分
    inst_streak_full: int = 5           # 連買 5 日 → 滿分
    # 籌碼結構
    margin_drop_full: float = 0.10      # 融資餘額 -10% → 滿分
    short_ratio_full: float = 0.30      # 券資比 30% → 滿分
    # 治理
    veto_score_cap: float = 20.0        # 13MV 下彎時總分上限（方法論鐵律）
    min_pe_hist: int = 60               # 估值百分位所需最少歷史樣本
