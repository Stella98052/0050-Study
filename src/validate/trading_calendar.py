# -*- coding: utf-8 -*-
"""交易日曆（定案 1）。

主索引：同批股票實際交易日之「聯集」（approximation）。
輔助校驗：讀取 TWSE 官方休市日表；凡聯集索引中缺漏、但既非週末亦非
官方休市日的日期，寫入 warnings，標示為「疑似資料缺漏而非假日」。

【已知限制，必須寫入報告 warnings】
若十檔股票在同一交易日全部缺漏，聯集法無法偵測該日，將被誤判為休市。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from config.config import Config

UNION_LIMITATION_WARNING = (
    "【已知限制】理論交易日曆採同批股票交易日聯集（approximation）："
    "若全部股票於同一交易日皆缺漏，該日無法被偵測，將被誤判為休市。"
)


def build_union_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """同批股票各自實際交易日之聯集，作為理論交易日曆（主索引）。"""
    all_dates: set[pd.Timestamp] = set()
    for df in frames.values():
        all_dates.update(pd.to_datetime(df["date"]).dt.normalize())
    return pd.DatetimeIndex(sorted(all_dates))


def _roc_or_ad_to_date(s: str) -> date | None:
    """TWSE 休市日表日期欄可能為民國或西元格式，統一轉 date；無法解析回傳 None。"""
    s = s.strip().replace("-", "/")
    parts = s.split("/")
    try:
        if len(parts) == 3:
            y = int(parts[0])
            y = y + 1911 if y < 1911 else y
            return date(y, int(parts[1]), int(parts[2]))
        if len(s) == 8 and s.isdigit():
            return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None
    return None


def _parse_holiday_payload(payload) -> set[date]:
    """相容兩種官方格式：
    ① OpenAPI：list of dicts，日期欄名 Date/date（可能民國或西元）
    ② 官網 rwd：{"stat":"OK","data":[[日期, 名稱, 說明,...], ...]}
    """
    holidays: set[date] = set()
    rows = payload if isinstance(payload, list) else (
        payload.get("data", []) if isinstance(payload, dict) else []
    )
    for row in rows:
        if isinstance(row, dict):
            raw = row.get("Date") or row.get("date") or ""
        elif isinstance(row, (list, tuple)) and row:
            raw = row[0]
        else:
            continue
        d = _roc_or_ad_to_date(str(raw))
        if d is not None:
            holidays.add(d)
    return holidays


def fetch_twse_holidays(
    cfg: Config, session: requests.Session | None = None
) -> set[date] | None:
    """抓取 TWSE 官方「有價證券集中交易市場開（休）市日期」。

    依 cfg.twse_holiday_urls 逐一嘗試（OpenAPI → 官網 rwd），任一成功即回傳
    並於 stdout 印出實際採用的來源（可追溯）。全部失敗回傳 None，由呼叫端
    降級：跳過輔助校驗並輸出明確警告，不可假裝已校驗、不可捏造假日。
    """
    sess = session or requests.Session()
    for url in cfg.twse_holiday_urls:
        try:
            resp = sess.get(
                url,
                timeout=cfg.request_timeout_sec,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            resp.raise_for_status()
            holidays = _parse_holiday_payload(resp.json())
            if holidays:
                print(f"[日曆] 官方休市日表來源：{url}（{len(holidays)} 筆）")
                return holidays
        except Exception:                            # noqa: BLE001（換下一候選）
            continue
    return None


def cross_check_calendar(
    calendar: pd.DatetimeIndex,
    holidays: set[date] | None,
    period_start: date,
    period_end: date,
) -> list[str]:
    """輔助校驗（定案 1）：找出「非週末、非官方休市日、卻不在聯集索引」的日期。

    重要限制（2026/7/12 實跑修正）：官方休市日表通常僅涵蓋近年度
    （實測 rwd 端點僅回當年度 27 筆），故校驗只在休市日表的實際涵蓋期間
    [min(holidays), max(holidays)] 內進行；範圍外的歷史假日無法辨識，
    一律不校驗並明確告知，不可將涵蓋範圍外的假日誤報為資料缺漏。
    holidays 為 None 時輸出降級警告；一律附上聯集法已知限制警語。
    """
    warnings: list[str] = [UNION_LIMITATION_WARNING]
    if holidays is None:
        warnings.append(
            "官方休市日表無法取得，輔助校驗已跳過（僅剩聯集法主索引）。"
            "缺漏日是否為假日無法自動判別，需人工確認。"
        )
        return warnings

    cover_start, cover_end = min(holidays), max(holidays)
    check_start = max(period_start, cover_start)
    check_end = min(period_end, cover_end)
    warnings.append(
        f"官方休市日表涵蓋 {cover_start} ~ {cover_end}（{len(holidays)} 筆），"
        f"輔助校驗僅於此範圍內進行；範圍外的歷史假日無法自動辨識，不校驗。"
        f"另注意：臨時休市（颱風等）不在年度表內，被標為疑似缺漏時需人工查證"
        f"（案例：2026-07-10 巴威颱風休市）。"
    )
    if check_start > check_end:
        warnings.append("休市日表涵蓋範圍與資料期間無交集，本次輔助校驗實際未檢查任何日期。")
        return warnings

    cal_set = {ts.date() for ts in calendar}
    suspects: list[date] = []
    for ts in pd.date_range(check_start, check_end, freq="D"):
        d = ts.date()
        if ts.weekday() >= 5:          # 週末
            continue
        if d in holidays or d in cal_set:
            continue
        suspects.append(d)
    if suspects:
        head = ", ".join(str(d) for d in suspects[:10])
        warnings.append(
            f"疑似資料缺漏而非假日：共 {len(suspects)} 個平日（於涵蓋範圍內）"
            f"不在聯集索引且非官方休市日（前 10 筆：{head}）。"
        )
    return warnings
