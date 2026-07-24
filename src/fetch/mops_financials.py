# -*- coding: utf-8 -*-
"""MOPS 季報彙總（v3.27）：EPS／三率／ROE／FCF。

來源（單季全市場一檔，與月營收同型）：
  綜合損益表彙總 https://mopsov.twse.com.tw/mops/web/ajax_t163sb04
  資產負債表彙總 https://mopsov.twse.com.tw/mops/web/ajax_t163sb05
  現金流量表彙總 https://mopsov.twse.com.tw/mops/web/ajax_t163sb20（待驗證）

**防前視（本模組最關鍵處）**：財報有法定公告期限，季底當天資料尚未
存在。本系統採**法定期限為生效日**（保守，實際多提早公告）：
  Q1→5/15、Q2→8/14、Q3→11/14、Q4(全年)→次年 3/31
交易日 t 只能使用「生效日 <= t」的季度。以季底日期對齊＝前視洩漏。

解析一律欄名驅動；失敗必印結構（L60）；快取綁解析器版本（L61）。
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.fetch.twse_daily import _TWSE_HEADERS, _atomic_to_csv

PARSER_VERSION = "v1"
_BASE = "https://mopsov.twse.com.tw/mops/web/"
_ENDPOINTS = {"income": "ajax_t163sb04", "balance": "ajax_t163sb05",
              "cashflow": "ajax_t163sb20"}

FIN_COLS = ["stock_id", "year", "season", "eps", "revenue", "gross_profit",
            "op_income", "net_income", "equity", "ocf", "capex"]

# 法定公告期限（防前視生效日）：(月, 日, 是否次年)
_DEADLINE = {1: (5, 15, False), 2: (8, 14, False),
             3: (11, 14, False), 4: (3, 31, True)}


def publication_date(year_ad: int, season: int) -> date:
    """該季財報之法定公告期限＝本系統採用的生效日（保守）。"""
    m, d, next_year = _DEADLINE[season]
    return date(year_ad + (1 if next_year else 0), m, d)


def available_quarters(asof: date, n: int = 8) -> list[tuple[int, int]]:
    """asof 當日「已可使用」的季度，新到舊，最多 n 個。"""
    out: list[tuple[int, int]] = []
    y, s = asof.year, 4
    while len(out) < n and y > asof.year - 6:
        if publication_date(y, s) <= asof:
            out.append((y, s))
        s -= 1
        if s == 0:
            y, s = y - 1, 4
    return out


def _norm(c) -> str:
    return "".join(str(c).split())


def _num(x) -> float:
    t = str(x).replace(",", "").strip()
    if t in ("", "-", "--", "nan", "None"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _pick(fields: list[str], *keys, exclude=()) -> int | None:
    for i, f in enumerate(fields):
        if all(k in f for k in keys) and not any(x in f for x in exclude):
            return i
    return None


def parse_financial_html(html_text: str, spec: dict) -> pd.DataFrame:
    """彙總表 HTML → DataFrame。spec: {輸出欄: (關鍵字…,)}，欄名驅動。

    各業別（一般業/金融業/證券業）表格欄名不同，故逐表掃描、
    取得含「公司代號」且命中至少一個目標欄的表格並縱向合併。
    """
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except ValueError:
        return pd.DataFrame(columns=["stock_id"] + list(spec))
    frames = []
    for t in tables:
        cols = [_norm(c) for c in t.columns]
        i_id = _pick(cols, "公司代號")
        if i_id is None:
            continue
        idx = {}
        for out_col, keys in spec.items():
            j = _pick(cols, *keys[0], exclude=keys[1] if len(keys) > 1 else ())
            if j is not None:
                idx[out_col] = j
        if not idx:
            continue
        rows = []
        for _, r in t.iterrows():
            sid = str(r.iloc[i_id]).strip()
            if not sid.isdigit():
                continue
            rec = {"stock_id": sid}
            for out_col, j in idx.items():
                rec[out_col] = _num(r.iloc[j])
            rows.append(rec)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["stock_id"] + list(spec))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["stock_id"]).reset_index(drop=True)


_SPEC_INCOME = {
    "eps": (("每股盈餘",), ("稀釋",)),
    "revenue": (("營業收入",), ()),
    "gross_profit": (("營業毛利",), ()),
    "op_income": (("營業利益",), ()),
    "net_income": (("本期淨利",), ("非控制", "綜合")),
}
_SPEC_BALANCE = {"equity": (("權益總",), ("非控制",))}
_SPEC_CASHFLOW = {
    "ocf": (("營業活動",), ()),
    "capex": (("取得不動產",), ()),
}


def describe_tables(html_text: str, limit: int = 3) -> str:
    """結構診斷：列出各表欄名（解析零列時定位用）。"""
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except ValueError as exc:
        return f"無表格：{exc}（len={len(html_text)}）"
    parts = [f"n_tables={len(tables)}"]
    for i, t in enumerate(tables[:limit]):
        parts.append(f"  table[{i}] shape={t.shape} "
                     f"cols={[_norm(c) for c in t.columns][:16]}")
    return "\n    ".join(parts)


def fetch_quarter(kind: str, year_ad: int, season: int, cfg,
                  session=None) -> pd.DataFrame:
    """單季全市場某一表（income/balance/cashflow）；快取＋失敗必印結構。"""
    spec = {"income": _SPEC_INCOME, "balance": _SPEC_BALANCE,
            "cashflow": _SPEC_CASHFLOW}[kind]
    cache = (Path("data/raw_fin") / f"{kind}_{PARSER_VERSION}"
             / f"{year_ad}Q{season}.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache, dtype={"stock_id": str})
    sess = session or requests.Session()
    data = {"encodeURIComponent": "1", "step": "1", "firstin": "1",
            "off": "1", "TYPEK": "sii", "year": str(year_ad - 1911),
            "season": f"{season:02d}"}
    last_err, df = None, pd.DataFrame()
    for attempt in range(3):
        try:
            resp = sess.post(_BASE + _ENDPOINTS[kind], data=data,
                             headers=_TWSE_HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            df = parse_financial_html(resp.text, spec)
            if len(df) == 0:
                last_err = ("解析零列，實際結構：\n    "
                            + describe_tables(resp.text))
            break
        except Exception as exc:                           # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    time.sleep(getattr(cfg, "request_delay_sec", 1.5))
    if len(df) == 0:
        print(f"  [FIN:{kind}] {year_ad}Q{season} 無資料｜{last_err}")
        return df
    df["year"], df["season"] = year_ad, season
    _atomic_to_csv(df, cache)
    return df


def fetch_financials(stock_id: str, asof: date, cfg, n_quarters: int = 6,
                     session=None) -> pd.DataFrame:
    """單股近 n 季財務（僅含已過公告期限者）。新到舊排序。"""
    recs = []
    for (y, s) in available_quarters(asof, n_quarters):
        row = {"stock_id": str(stock_id), "year": y, "season": s,
               "pub_date": publication_date(y, s)}
        for kind in ("income", "balance", "cashflow"):
            tbl = fetch_quarter(kind, y, s, cfg, session)
            if len(tbl) == 0:
                continue
            sub = tbl[tbl["stock_id"].astype(str) == str(stock_id)]
            if len(sub) == 0:
                continue
            for c in sub.columns:
                if c not in ("stock_id", "year", "season"):
                    row[c] = float(sub.iloc[0][c]) if pd.notna(
                        sub.iloc[0][c]) else float("nan")
        recs.append(row)
    return pd.DataFrame(recs)
