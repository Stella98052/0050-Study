# -*- coding: utf-8 -*-
"""VG-1 資料完整性關卡。

檢查項目：欄位型別、重複日期、負值（價/量）、日期連續性（缺漏區間）、
完整率 = 實際交易日數 / 理論交易日數；低於門檻（Config 預設 0.95）
→ passed=False 且 warnings 列出缺漏區間。呼叫端必須將 warnings 印出，
不可讓使用者誤以為已驗證通過。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from config.config import Config, DISCLAIMER
from src.schemas import OHLCV_SCHEMA, VG1Report


def _check_dtypes(df: pd.DataFrame) -> bool:
    """欄位存在且型別可安全視為符合 OHLCV_SCHEMA。"""
    for col, want in OHLCV_SCHEMA.items():
        if col not in df.columns:
            return False
        got = str(df[col].dtype)
        if want == "object":
            ok = got in ("object", "string", "str")
        elif want.startswith("datetime64"):
            ok = got.startswith("datetime64")
        elif want == "int64":
            ok = got in ("int64", "int32", "Int64")
        else:
            ok = got in ("float64", "float32", "Float64")
        if not ok:
            return False
    return True


def _missing_ranges(
    missing: pd.DatetimeIndex,
) -> tuple[tuple[str, str], ...]:
    """將缺漏日期壓縮為連續區間 [(start, end), ...]（ISO 字串）。"""
    if len(missing) == 0:
        return ()
    ranges: list[tuple[str, str]] = []
    start = prev = missing[0]
    for ts in missing[1:]:
        if (ts - prev).days > 7:      # 相隔逾一週視為不同缺漏區間
            ranges.append((start.date().isoformat(), prev.date().isoformat()))
            start = ts
        prev = ts
    ranges.append((start.date().isoformat(), prev.date().isoformat()))
    return tuple(ranges)


def run_vg1_validation(
    df: pd.DataFrame,
    stock_id: str,
    calendar: pd.DatetimeIndex,
    cfg: Config,
    extra_warnings: tuple[str, ...] = (),
) -> VG1Report:
    """執行 VG-1 全部檢查並回傳結構化報告。

    calc_logic：
        completeness_rate = 該股實際交易日數 / 理論交易日數（聯集日曆在
        該股期間內的日數）；缺漏日 = 日曆有、該股沒有的日期。
    extra_warnings：交易日曆模組之輔助校驗結果（定案 1），一併寫入報告。
    """
    warnings: list[str] = list(extra_warnings)

    dtype_ok = _check_dtypes(df)
    if not dtype_ok:
        warnings.append("欄位型別不符 OHLCV_SCHEMA，需檢查抓取/解析層。")

    dates = pd.to_datetime(df["date"]).dt.normalize()
    n_dup = int(dates.duplicated().sum())
    if n_dup:
        warnings.append(f"重複日期 {n_dup} 筆。")

    price_cols = ["open", "high", "low", "close"]
    n_neg = int((df[price_cols] < 0).sum().sum() + (df["volume"] < 0).sum())
    if n_neg:
        warnings.append(f"價格或成交量負值 {n_neg} 筆。")

    if len(df):
        p_start, p_end = dates.min(), dates.max()
        cal_in_period = calendar[(calendar >= p_start) & (calendar <= p_end)]
        have = set(dates)
        missing = pd.DatetimeIndex([d for d in cal_in_period if d not in have])
        theoretical = len(cal_in_period)
        actual = int(dates.nunique())
        rate = actual / theoretical if theoretical else 0.0
    else:
        p_start = p_end = pd.Timestamp("1970-01-01")
        missing = pd.DatetimeIndex([])
        theoretical, actual, rate = 0, 0, 0.0
        warnings.append("資料為空。")

    ranges = _missing_ranges(missing)
    if rate < cfg.completeness_warn_threshold:
        warnings.append(
            f"完整率 {rate:.2%} 低於門檻 {cfg.completeness_warn_threshold:.0%}，"
            f"缺漏區間：{list(ranges)}。不可靜默接受，需人工確認。"
        )

    passed = (
        dtype_ok
        and n_dup == 0
        and n_neg == 0
        and rate >= cfg.completeness_warn_threshold
        and len(df) > 0
    )
    return VG1Report(
        stock_id=stock_id,
        period_start=p_start.date().isoformat(),
        period_end=p_end.date().isoformat(),
        n_rows=int(len(df)),
        dtype_ok=dtype_ok,
        n_duplicate_dates=n_dup,
        n_negative_values=n_neg,
        theoretical_trading_days=theoretical,
        actual_trading_days=actual,
        completeness_rate=round(rate, 6),
        missing_ranges=ranges,
        passed=passed,
        warnings=tuple(warnings),
    )


def save_validation_report(report: VG1Report, out_dir: Path) -> Path:
    """序列化為 data/validation_report_{stock_id}.json（開頭附風險聲明）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"validation_report_{report.stock_id}.json"
    payload = {"disclaimer": DISCLAIMER, **dataclasses.asdict(report)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_report(report: VG1Report) -> None:
    """將檢查結果與全部 warnings 印至 stdout（強制，不可靜默）。"""
    status = "通過" if report.passed else "未通過（見警告）"
    print(f"[VG-1] {report.stock_id}: {status}  完整率={report.completeness_rate:.2%}")
    for w in report.warnings:
        print(f"  ⚠ {w}")
