# -*- coding: utf-8 -*-
"""第二階段參數（與 phase1 Config 併用）。集中管理，禁止散落硬編碼。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PHASE2_VERSION: str = "2.16"


@dataclass(frozen=True)
class Phase2Config:
    # ---- 模型（定案：LightGBM）----
    model_backend: Literal["lightgbm", "xgboost"] = "lightgbm"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    random_state: int = 42

    # ---- 標籤（定案：淨報酬>0；進場=訊號日下一交易日開盤，出場=第N日收盤）----
    forward_return_days: int = 5
    label_rule: Literal["net_return_positive"] = "net_return_positive"

    # ---- Walk-Forward ----
    train_window_years: int = 3
    test_window_months: int = 3
    step_months: int = 3
    embargo_days: int = 30              # 定案：≥ MACD慢線26 取整30
    holdout_months: int = 12            # 完全獨立樣本外，不參與任何折/敏感度

    # ---- 敏感度分析（樣本內，明確標註）----
    zigzag_grid: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)

    # ---- 技術特徵窗口 ----
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ---- VG 門檻 ----
    min_independent_signals: int = 30
    permutation_n: int = 500
    bootstrap_n: int = 1000
    confidence_level: float = 0.95
    vg2_control_mode: Literal["random_listed_10"] = "random_listed_10"
    vg2_random_seed: int = 42

    # ---- 產出 ----
    model_dir: Path = Path("data/models")
    report_dir: Path = Path("data/reports")
    bundle_version_tag: str = "phase2-v1"
