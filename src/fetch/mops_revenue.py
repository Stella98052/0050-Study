# -*- coding: utf-8 -*-
"""MOPS 官方月營收（P2 第二項）。

來源：https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{民國年}_{月}_0.html
（上市全市場單月一檔；十年約 121 檔，快取於 data/raw_revenue/YYYYMM.csv）

防前視（法定發布時滯）：M 月營收依法於次月 10 日前公告——本系統採
保守生效日 = 次月 10 日；交易日 t 只能使用生效日 <= t 的最新月營收。
YoY = 當期月營收 / 12 個月前月營收 − 1。
治理：失敗必印原因、空結果不快取（L40-L42 全套沿用）。"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.fetch.twse_daily import _TWSE_HEADERS, _atomic_to_csv, month_range

_URL_TPL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{y}_{m}_0.html"
_CSV_URL = "https://mopsov.twse.com.tw/server-java/FileDownLoad"
REV_COLS = ["stock_id", "revenue"]


def _norm(c) -> str:
    """欄名正規化：攤平多層、去除所有空白（L44：表頭含 <br> 攤平後帶
    空白致關鍵字比對失手，如「公司 代號」）。"""
    t = "".join(map(str, c)) if isinstance(c, tuple) else str(c)
    return "".join(t.split())


def parse_revenue_csv(content: bytes) -> pd.DataFrame:
    """官方『另存CSV』檔 → [stock_id, revenue]。編碼 cp950(big5) 優先。

    實際表頭例：出表日期,公司代號,公司名稱,產業別,營業收入-當月營收,
    營業收入-上月營收,營業收入-去年當月營收,…（欄名驅動同 HTML 版）。"""
    for enc in ("cp950", "big5hkscs", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, dtype=str)
            break
        except Exception:                                 # noqa: BLE001
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=REV_COLS)
    cols = [_norm(c) for c in df.columns]
    i_id = next((i for i, c in enumerate(cols) if "公司代號" in c), None)
    i_rev = next((i for i, c in enumerate(cols)
                  if "當月營收" in c and "去年" not in c), None)
    if i_id is None or i_rev is None:
        return pd.DataFrame(columns=REV_COLS)
    out = []
    for _, r in df.iterrows():
        sid = str(r.iloc[i_id]).strip()
        if not sid.isdigit():
            continue
        rev = pd.to_numeric(str(r.iloc[i_rev]).replace(",", ""),
                            errors="coerce")
        if pd.notna(rev):
            out.append({"stock_id": sid, "revenue": float(rev)})
    return pd.DataFrame(out, columns=REV_COLS).drop_duplicates(
        subset=["stock_id"]).reset_index(drop=True)


def fetch_revenue_csv(ym: str, sess, timeout: int) -> pd.DataFrame:
    """官方 CSV 下載（頁面『另存CSV』的 POST 表單）。"""
    y, m = int(ym[:4]) - 1911, int(ym[4:])
    data = {"step": "9", "functionName": "show_file2",
            "filePath": "/t21/sii/", "fileName": f"t21sc03_{y}_{m}.csv"}
    resp = sess.post(_CSV_URL, data=data, headers=_TWSE_HEADERS,
                     timeout=timeout)
    resp.raise_for_status()
    return parse_revenue_csv(resp.content)


def parse_revenue_html(html_text: str) -> pd.DataFrame:
    """全市場月報 HTML → [stock_id, revenue]（當月營收，仟元）。

    欄名驅動（L41）：掃所有表格，找同時含「公司代號」與「當月營收」
    語意欄者；多層表頭攤平後比對關鍵字。"""
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except ValueError:
        return pd.DataFrame(columns=REV_COLS)
    out_rows = []
    for t in tables:
        cols = [_norm(c) for c in t.columns]
        i_id = next((i for i, c in enumerate(cols) if "公司代號" in c), None)
        i_rev = next((i for i, c in enumerate(cols)
                      if "當月營收" in c and "去年" not in c), None)
        if i_id is None or i_rev is None:
            continue
        for _, r in t.iterrows():
            sid = str(r.iloc[i_id]).strip()
            if not sid.isdigit():
                continue
            rev = pd.to_numeric(str(r.iloc[i_rev]).replace(",", ""),
                                errors="coerce")
            if pd.notna(rev):
                out_rows.append({"stock_id": sid, "revenue": float(rev)})
    df = pd.DataFrame(out_rows, columns=REV_COLS)
    return df.drop_duplicates(subset=["stock_id"]).reset_index(drop=True)


def fetch_revenue_month(ym: str, cfg,
                        session: requests.Session | None = None
                        ) -> pd.DataFrame:
    """單月全市場營收（快取優先；空結果不快取並印原因）。"""
    sess = session or requests.Session()
    cache_dir = Path(getattr(cfg, "cache_dir", Path("data/raw"))).parent \
        / "raw_revenue"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{ym}.csv"
    if cpath.exists():
        return pd.read_csv(cpath, dtype={"stock_id": str})
    y, m = int(ym[:4]) - 1911, int(ym[4:])
    url = _URL_TPL.format(y=y, m=m)
    last_err = None
    df = pd.DataFrame(columns=REV_COLS)
    for attempt in range(3):
        try:
            df = fetch_revenue_csv(ym, sess, 25)          # 主：官方CSV
            if len(df):
                break
            last_err = "CSV 解析零列"
            resp = sess.get(url, headers=_TWSE_HEADERS, timeout=25)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "big5"
            df = parse_revenue_html(resp.text)            # 備：HTML
            if len(df) == 0:
                last_err += f"；HTML 亦零列（len={len(resp.text)}）"
            break
        except Exception as exc:                          # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    if len(df) == 0:
        print(f"  [REV] {ym} 無資料｜{last_err}")
        return df
    _atomic_to_csv(df, cpath)
    time.sleep(getattr(cfg, "request_delay_sec", 1.5))
    return df


def build_revenue_map(stock_ids, start: date, end: date, cfg,
                      session=None) -> dict[str, dict[str, float]]:
    """{stock_id: {yyyymm: revenue}}。需多抓 13 個月供 YoY 基期。"""
    months = month_range(start, end)
    # 往前補 13 個月
    y0, m0 = int(months[0][:4]), int(months[0][4:])
    extra = []
    for k in range(13, 0, -1):
        mm = m0 - k
        yy = y0
        while mm <= 0:
            mm += 12
            yy -= 1
        extra.append(f"{yy}{mm:02d}")
    out: dict[str, dict[str, float]] = {s: {} for s in stock_ids}
    for ym in extra + months:
        tbl = fetch_revenue_month(ym, cfg, session)
        if len(tbl) == 0:
            continue
        sub = tbl[tbl["stock_id"].isin(stock_ids)]
        for _, r in sub.iterrows():
            out[r["stock_id"]][ym] = float(r["revenue"])
    return out


def _prev_ym(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[4:])
    m -= k
    while m <= 0:
        m += 12
        y -= 1
    return f"{y}{m:02d}"


def revenue_yoy_on_dates(dates: pd.Series,
                         rev_by_month: dict[str, float]) -> pd.Series:
    """逐交易日之可用月營收 YoY（發布日對齊：M 月營收自次月10日起可用）。

    交易日 t：最新可用月份 M = （t.day>=10 ? t 前一月 : t 前二月）；
    yoy = rev[M]/rev[M-12] − 1；任一缺 → NaN。"""
    d = pd.to_datetime(dates)
    out = []
    for t in d:
        lag = 1 if t.day >= 10 else 2
        m_ = _prev_ym(f"{t.year}{t.month:02d}", lag)
        cur, base = rev_by_month.get(m_), rev_by_month.get(_prev_ym(m_, 12))
        out.append(cur / base - 1.0
                   if cur is not None and base not in (None, 0) else float("nan"))
    return pd.Series(out, index=dates.index, dtype="float64")


def revenue_yoy_latest(stock_id: str, asof: date, cfg,
                       session=None) -> tuple[float | None, str]:
    """最新可用月營收 YoY——僅抓 2 個月報檔（面板即時用，避免 40 秒等待）。

    發布日對齊同 revenue_yoy_on_dates：asof.day>=10 → 可用前一月，
    否則前二月。若該月檔尚未產生（月初），自動再往前一個月。
    回傳 (yoy 或 None, 說明字串)。
    """
    ym0 = f"{asof.year}{asof.month:02d}"
    lag = 1 if asof.day >= 10 else 2
    for extra in (0, 1):                                   # 最多退一個月
        m = _prev_ym(ym0, lag + extra)
        base = _prev_ym(m, 12)
        cur_tbl = fetch_revenue_month(m, cfg, session)
        if len(cur_tbl) == 0:
            continue
        base_tbl = fetch_revenue_month(base, cfg, session)
        if len(base_tbl) == 0:
            return None, f"{base} 基期月報無資料"
        cur = cur_tbl[cur_tbl["stock_id"] == str(stock_id)]["revenue"]
        bas = base_tbl[base_tbl["stock_id"] == str(stock_id)]["revenue"]
        if len(cur) == 0 or len(bas) == 0 or float(bas.iloc[0]) == 0:
            return None, f"{stock_id} 於 {m}/{base} 無營收列"
        yoy = float(cur.iloc[0]) / float(bas.iloc[0]) - 1.0
        return yoy, f"{m} 營收 YoY（基期 {base}）"
    return None, "最近月報尚未發布"
