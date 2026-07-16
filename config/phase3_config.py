# -*- coding: utf-8 -*-
"""第三階段設定（面板/每日更新/前瞻紀錄）。全域參數集中，禁散落硬編碼。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

PHASE3_VERSION = "3.10"


@dataclass(frozen=True)
class Phase3Config:
    chart_lookback_days: int = 250            # 面板K線顯示根數
    model_path: Path = Path("data/models/model_phase2-v1.joblib")
    report_json: Path = Path("data/reports/phase2_report.json")
    predictions_csv: Path = Path("data/predictions.csv")
    min_prospective_samples: int = 30         # Model v2 前瞻最終閘（預先宣告）
    disclaimer_short: str = ("非投資建議、歷史績效不代表未來表現。"
                             "本系統僅供研究與教育用途。")
