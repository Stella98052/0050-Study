# -*- coding: utf-8 -*-
"""全域設定 Config — 系統唯一參數來源，禁止任何模組硬編碼魔法數字。

風險聲明（固定文字，所有報告開頭引用 DISCLAIMER）：
    本系統僅供研究與教育用途，歷史回測績效不代表未來表現，不構成投資建議。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PHASE1_VERSION: str = "1.6"   # 版本橫幅：啟動時列印，防止版本混用（LESSONLEARNT）

DISCLAIMER: str = (
    "【風險聲明】本系統僅供研究與教育用途，歷史回測績效不代表未來表現，"
    "不構成投資建議。波浪理論屬主觀分析框架，本系統採規則化近似"
    "（approximation），非唯一標準。"
)


@dataclass(frozen=True)
class Config:
    """全域參數。

    交易成本預設值來源：臺灣證券交易所現行規定 —
    券商手續費上限 0.1425%（買賣各計一次）、證券交易稅 0.3%（賣出課徵）。
    """

    # ---- 資料範圍 ----
    history_years: int = 10
    expected_holdings_count: int = 10          # 0050 前十大持股數（VG-1 檢查）

    # ---- TWSE 抓取行為 ----
    twse_stock_day_url: str = (
        "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    )
    twse_holiday_urls: tuple[str, ...] = (
        "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule",
        "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule?response=json",
    )
    request_delay_sec_min: float = 1.0
    request_delay_sec_max: float = 2.0
    max_retries: int = 3
    retry_backoff_base_sec: float = 2.0
    request_timeout_sec: float = 15.0
    cache_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    failure_log_path: Path = Path("logs/fetch_failures.log")

    # ---- VG-1 資料完整性 ----
    completeness_warn_threshold: float = 0.95

    # ---- ZigZag / 波浪 ----
    zigzag_threshold: float = 0.05
    # 閾值越小 → 轉折越多、波段越碎、雜訊越高；
    # 閾值越大 → 波段越平滑，但 realtime 確認延遲越久。
    zigzag_sensitivity_min: float = 0.03       # 第二階段敏感度分析下限
    zigzag_sensitivity_max: float = 0.08       # 上限（樣本內優化，需留獨立樣本外）

    # ---- 推進浪 / 修正浪量化判別（approximation）----
    impulse_min_amplitude_ratio: float = 1.0   # 推進浪幅度 ≥ 相鄰修正浪幅度 × 此值
    impulse_min_duration_bars: int = 2         # 波段最少 K 棒數，低於此列 unknown
    fib_levels: tuple[float, ...] = (0.382, 0.5, 0.618)
    fib_tolerance: float = 0.05                # 斐波那契命中容忍（±5%），僅輔助標記

    # ---- MV 潮汐量能（定案 2：5 / 13 / 20 三線）----
    vol_ma_short: int = 5                      # 5MV 波段攻擊量
    vol_ma_mid: int = 13                       # 13MV：方法論核心否決線（絕對否決）
    vol_ma_long: int = 20                      # 20MV 規格長均線（乖離率分母）
    core_veto_line: Literal["mv_mid"] = "mv_mid"   # 定案 2：13MV 標記為核心否決線
    volume_burst_bias_threshold: float = 0.20  # 量能乖離率 > 20% = 潮汐爆發

    # ---- 標籤 / 報酬定義（第二階段用，先行集中定義）----
    forward_return_days: int = 5               # 未來 N 天報酬（亦為定案 4.2 之 N）
    entry_price_rule: Literal["next_open"] = "next_open"

    # ---- 交易成本（來源見 docstring；寫死預設值、可參數覆寫）----
    fee_buy_rate: float = 0.001425
    fee_sell_rate: float = 0.001425
    tax_sell_rate: float = 0.003

    # ---- 統計驗證（VG-3 / VG-4）----
    min_independent_signals: int = 30
    permutation_n_iterations: int = 500
    bootstrap_n_iterations: int = 1000
    confidence_level: float = 0.95
    random_state: int = 42
