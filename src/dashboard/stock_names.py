# -*- coding: utf-8 -*-
"""代號→名稱對照（面板下拉顯示用；治理：名稱亦須可追溯官方來源，禁手寫表）。

來源優先序：
  ① holdings.csv 的 name 欄（與持股清單同源同基準日，最一致）
  ② TWSE MIS 即時報價的 name 欄（官方，SmartTWFetcher 既有）
  ③ 皆無 → 只回代號本身（不捏造名稱）
"""
from __future__ import annotations
import csv
from pathlib import Path


def load_names_from_holdings(path: Path) -> dict[str, str]:
    """讀 holdings.csv 的 stock_id→name（無 name 欄則回空 dict，不報錯）。"""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for r in rows:
        sid = str(r.get("stock_id", "")).strip()
        name = str(r.get("name", "")).strip()
        if sid and name:
            out[sid] = name
    return out


def format_choice(stock_id: str, names: dict[str, str]) -> str:
    """下拉顯示字串：有名稱→『2330 台積電』；無→『2330』（不捏造）。"""
    nm = names.get(stock_id)
    return f"{stock_id} {nm}" if nm else stock_id


_T187_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
_NAME_KEYS = ("公司簡稱", "公司名稱", "Name", "companyName")
_CODE_KEYS = ("公司代號", "Code", "companyCode")


def parse_company_names(rows: list[dict]) -> dict[str, str]:
    """t187ap03_L JSON → {代號: 簡稱}（欄名驅動；簡稱優先、名稱備援）。"""
    out: dict[str, str] = {}
    for r in rows or []:
        sid = next((str(r[k]).strip() for k in _CODE_KEYS if k in r), "")
        nm = next((str(r[k]).strip() for k in _NAME_KEYS
                   if k in r and str(r[k]).strip()), "")
        if sid.isdigit() and nm:
            out[sid] = nm
    return out


def fetch_all_listed_names(timeout: int = 15) -> dict[str, str]:
    """官方全市場代號→簡稱（自選股名稱用）；失敗回空 dict 不擋面板。"""
    import requests
    try:
        r = requests.get(_T187_URL, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        r.raise_for_status()
        return parse_company_names(r.json())
    except Exception:                                     # noqa: BLE001
        return {}
