# -*- coding: utf-8 -*-
"""TWSE 官方日線抓取（STOCK_DAY，按月查詢）。不可使用 yfinance。

資料來源：https://www.twse.com.tw/exchangeReport/STOCK_DAY
回傳格式（官方 JSON）：
    fields: ["日期","成交股數","成交金額","開盤價","最高價","最低價",
             "收盤價","漲跌價差","成交筆數"]
    data:   [["113/09/02","35,000,000",...], ...]   # 民國年、千分位逗號
"""

from __future__ import annotations

import random
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

# TWSE 端點會拒絕無瀏覽器標頭的請求（回 403 Forbidden），故所有請求帶 UA。
# 這是抓取層對「TWSE 反爬」的必要相容，非偽裝——僅讀取官方公開日線資料。
_TWSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.twse.com.tw/",
}

from config.config import Config
from src.schemas import OHLCV_SCHEMA, FetchError

# STOCK_DAY 回傳欄位的固定位置（依官方 fields 順序）
_COL_DATE, _COL_VOLUME, _COL_OPEN, _COL_HIGH, _COL_LOW, _COL_CLOSE = 0, 1, 3, 4, 5, 6



def _atomic_to_csv(df, path) -> None:
    """v2.16 原子寫入（外部審查唯一採納項）：先寫同目錄暫存檔再
    os.replace 原子替換——中斷/當機不再留下半寫的髒快取
    （TWSE 錯月快取事件的同族風險）。"""
    import os
    import tempfile
    d = os.path.dirname(str(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def _roc_date_to_ad(roc: str) -> date:
    """民國年日期字串（例 '113/09/02'）轉西元 date。"""
    y, m, d = roc.strip().split("/")
    return date(int(y) + 1911, int(m), int(d))


def _to_float(s: str) -> float:
    """去除千分位逗號後轉 float；'--' 等無效值轉 NaN。"""
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _to_int(s: str) -> int:
    """去除千分位逗號後轉 int；無效值轉 -1（由 VG-1 負值檢查攔截）。"""
    s = s.replace(",", "").strip()
    try:
        return int(float(s))
    except ValueError:
        return -1


def parse_stock_day_json(payload: dict, stock_id: str) -> pd.DataFrame:
    """解析 STOCK_DAY 官方 JSON 為符合 OHLCV_SCHEMA 之 DataFrame。

    calc_logic：逐列取固定欄位位置（依官方 fields 順序），民國年轉西元、
    千分位轉數值；stat != 'OK' 視為該月無資料，回傳空表（由上層決定是否警告）。
    """
    if payload.get("stat") != "OK" or not payload.get("data"):
        return _empty_frame()
    rows = []
    for r in payload["data"]:
        rows.append(
            {
                "stock_id": stock_id,
                "date": pd.Timestamp(_roc_date_to_ad(r[_COL_DATE])),
                "open": _to_float(r[_COL_OPEN]),
                "high": _to_float(r[_COL_HIGH]),
                "low": _to_float(r[_COL_LOW]),
                "close": _to_float(r[_COL_CLOSE]),
                "volume": _to_int(r[_COL_VOLUME]),
            }
        )
    df = pd.DataFrame(rows)
    return df.astype(OHLCV_SCHEMA)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in OHLCV_SCHEMA.items()})


def _cache_path(stock_id: str, yyyymm: str, cfg: Config) -> Path:
    return cfg.cache_dir / stock_id / f"{yyyymm}.csv"


def _log_failure(stock_id: str, yyyymm: str, err: str, cfg: Config) -> None:
    """失敗記錄檔：時間戳 + 股票 + 月份 + 錯誤訊息（供 list_failed_months 重跑）。"""
    cfg.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.failure_log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{stock_id}\t{yyyymm}\t{err}\n")


def _keep_requested_month(df: pd.DataFrame, stock_id: str, yyyymm: str) -> pd.DataFrame:
    """防護：只保留屬於請求月份的資料列。

    背景（2026/7/11 實跑發現 2454/2308 重複日期各 ~20 筆 ≈ 恰一個月交易日，
    懷疑某月快取檔含了不屬於該月的列）。若偵測到越月列，明確印出警示
    （股票/月份/筆數），不靜默丟棄。
    """
    if len(df) == 0:
        return df
    y, m = int(yyyymm[:4]), int(yyyymm[4:])
    in_month = (df["date"].dt.year == y) & (df["date"].dt.month == m)
    n_out = int((~in_month).sum())
    if n_out:
        print(f"  ⚠ [{stock_id} {yyyymm}] 回應含 {n_out} 筆非該月資料列，已過濾"
              f"（governance：資料異常需可見，不靜默）")
    return df.loc[in_month].reset_index(drop=True)


def fetch_stock_month(
    stock_id: str,
    yyyymm: str,
    cfg: Config,
    session: requests.Session | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """抓取單一股票單月日線。歷史月份：快取命中即讀檔；未命中則發請求並寫入快取。

    當月例外（2026/7/12 修復）：進行中的月份資料每日都在增長，若沿用快取
    會永遠停在首次抓取日（例：7/11 快取的 202607 永遠缺 7/13 之後的交易日）。
    故 yyyymm == 本月 時一律重抓並覆寫快取；月份結束後自然轉為固定快取。

    重試：最多 max_retries 次，指數退避（2, 4, 8 秒）；每次請求前
    隨機延遲 1–2 秒（符合 TWSE 合理使用頻率）。耗盡後寫失敗記錄並 raise。
    無論來自快取或新請求，一律套用越月列過濾防護。
    """
    now = today or date.today()
    is_current_month = (yyyymm == f"{now.year}{now.month:02d}")
    cpath = _cache_path(stock_id, yyyymm, cfg)
    if cpath.exists() and not is_current_month:
        df = pd.read_csv(cpath, parse_dates=["date"])
        if len(df) == 0:
            return _empty_frame()
        return _keep_requested_month(df.astype(OHLCV_SCHEMA), stock_id, yyyymm)

    sess = session or requests.Session()
    params = {"response": "json", "date": f"{yyyymm}01", "stockNo": stock_id}
    last_err = ""
    for attempt in range(cfg.max_retries):
        time.sleep(random.uniform(cfg.request_delay_sec_min, cfg.request_delay_sec_max))
        try:
            resp = sess.get(
                cfg.twse_stock_day_url, params=params,
                headers=_TWSE_HEADERS, timeout=cfg.request_timeout_sec
            )
            resp.raise_for_status()
            df = parse_stock_day_json(resp.json(), stock_id)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            _atomic_to_csv(df, cpath)             # 空表亦快取；原子寫入防髒檔
            return _keep_requested_month(df, stock_id, yyyymm)
        except Exception as exc:                    # noqa: BLE001（記錄後重試）
            last_err = repr(exc)
            time.sleep(cfg.retry_backoff_base_sec * (2 ** attempt))
    _log_failure(stock_id, yyyymm, last_err, cfg)
    raise FetchError(f"{stock_id} {yyyymm} 抓取失敗（已重試 {cfg.max_retries} 次）：{last_err}")


def month_range(start: date, end: date) -> list[str]:
    """回傳 start~end 之間所有 yyyymm 字串（含頭尾月份）。"""
    months: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return months


def fetch_stock_history(
    stock_id: str,
    start: date,
    end: date,
    cfg: Config,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """逐月迴圈抓取並合併為完整日線（近 10 年）。

    本函式不吞掉缺漏月份：單月失敗即向上 raise FetchError；
    合併結果需立即交由 run_vg1_validation 檢查，不可靜默接受不完整資料。
    """
    frames = [
        fetch_stock_month(stock_id, mm, cfg, session=session)
        for mm in month_range(start, end)
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.sort_values("date").reset_index(drop=True)

    # 第二層防護：合併後仍有重複日期 → 明確警示並去重（保留首筆），不靜默
    dup_mask = df["date"].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        dup_dates = df.loc[dup_mask, "date"].dt.date.unique()
        head = ", ".join(str(d) for d in dup_dates[:5])
        print(f"  ⚠ [{stock_id}] 合併後偵測到重複日期 {n_dup} 筆（前5: {head}），"
              f"已去重保留首筆。請執行 diagnose_duplicates.py 找出來源快取檔。")
        df = df.loc[~dup_mask].reset_index(drop=True)
    return df


def list_failed_months(cfg: Config) -> list[tuple[str, str]]:
    """讀取失敗記錄檔，回傳 (stock_id, yyyymm) 清單，供重跑與人工確認。"""
    if not cfg.failure_log_path.exists():
        return []
    out: list[tuple[str, str]] = []
    for line in cfg.failure_log_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append((parts[1], parts[2]))
    return out
