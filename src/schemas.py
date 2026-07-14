# -*- coding: utf-8 -*-
"""共用資料模型（schema / dataclass）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

# 日線 OHLCV 統一 schema：所有抓取輸出必須符合。
# 日期一律 datetime64[ns]（西元）；成交量單位為「股」，不換算張數。
OHLCV_SCHEMA: dict[str, str] = {
    "stock_id": "object",
    "date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}

WaveLabelValue = Literal["1", "2", "3", "4", "5", "A", "B", "C", "unknown"]
LabelVersion = Literal["retrospective", "realtime"]


class FetchError(RuntimeError):
    """抓取在重試耗盡後仍失敗（已寫入失敗記錄檔）。"""


class HoldingsUnavailableError(RuntimeError):
    """0050 持股清單所有白名單來源皆不可用。此欄位缺失，需人工確認。"""


@dataclass(frozen=True)
class HoldingsSnapshot:
    """0050 前十大持股清單快照。

    來源白名單：MOPS（mops.twse.com.tw）、TWSE OpenAPI（openapi.twse.com.tw）、
    或使用者提供之 CSV 覆寫（is_manual_override=True）。
    禁止捏造：所有來源失敗且無 CSV 時必須 raise，不可回傳內建預設清單。
    """

    stock_ids: tuple[str, ...]
    snapshot_date: date
    source: str
    is_manual_override: bool


@dataclass(frozen=True)
class VG1Report:
    """VG-1 資料完整性關卡輸出（序列化為 validation_report_{stock_id}.json）。"""

    stock_id: str
    period_start: str                    # ISO 日期字串（利於 JSON 序列化）
    period_end: str
    n_rows: int
    dtype_ok: bool
    n_duplicate_dates: int
    n_negative_values: int
    theoretical_trading_days: int
    actual_trading_days: int
    completeness_rate: float
    missing_ranges: tuple[tuple[str, str], ...]
    passed: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Pivot:
    """ZigZag 轉折點。

    realtime 版本：confirmed_date 為「價格自極值反向移動達閾值」之當日，
    必然晚於 pivot_date（此延遲即防未來函數的代價）。
    retrospective 版本：confirmed_date=None（回溯視角無確認概念）。
    """

    bar_index: int
    pivot_date: date
    price: float
    kind: Literal["peak", "trough"]
    confirmed_date: Optional[date]


@dataclass(frozen=True)
class WaveSegment:
    """一段波浪（兩個相鄰轉折點之間）與其標籤及規則檢查結果。"""

    start_pivot: Pivot
    end_pivot: Pivot
    label: WaveLabelValue
    version: LabelVersion
    segment_type: Literal["impulse", "corrective", "unknown"]
    iron_rules: dict[str, bool] = field(default_factory=dict)
    fib_retracement_hit: Optional[float] = None
