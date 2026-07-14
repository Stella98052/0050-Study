# -*- coding: utf-8 -*-
"""產業分類（第四優先用；L21：先驗一律凍結 TWSE 官方分類，禁手寫表）。

來源①：TWSE OpenAPI t187ap03_L（上市公司基本資料，含產業別）
來源②：使用者 CSV 覆寫 industry_map.csv（欄位 stock_id,industry,source）
兩者皆不可用 → 明確 raise 並指引，不得回傳任何內建對照表（禁捏造）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import requests

_T187AP03_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
# 官方欄位名可能為中文或英文，雙鍵防禦
_CODE_KEYS = ("公司代號", "Code", "companyCode")
_IND_KEYS = ("產業別", "SecuritiesIndustryCode", "industry")


def fetch_industry_map(timeout: float = 25.0,
                       session=None) -> dict[str, str] | None:
    """自 TWSE 官方抓取全上市產業對照；不可用回傳 None（由呼叫端走 CSV）。"""
    sess = session or requests.Session()
    try:
        resp = sess.get(_T187AP03_URL, timeout=timeout,
                        headers={"Accept": "application/json",
                                 "User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        rows = resp.json()
    except Exception:                                    # noqa: BLE001
        return None
    if not isinstance(rows, list):
        return None
    out: dict[str, str] = {}
    for r in rows:
        code = next((str(r[k]).strip() for k in _CODE_KEYS if k in r), None)
        ind = next((str(r[k]).strip() for k in _IND_KEYS if k in r), None)
        if code and ind:
            out[code] = ind
    return out if len(out) >= 100 else None              # 太少視為解析失敗


def load_industry_csv(path: Path) -> dict[str, str]:
    """人工 fallback：CSV（stock_id,industry,source），來源需註明。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {str(r["stock_id"]).strip(): str(r["industry"]).strip()
           for r in rows if str(r.get("stock_id", "")).strip()}
    if not out:
        raise ValueError(f"{path} 無有效對照列")
    return out


def resolve_industry_map(stock_ids, csv_path: Path | None = None
                         ) -> tuple[dict[str, str], str]:
    """官方優先 → CSV fallback → raise。回傳 (對照, 來源說明)。"""
    m = fetch_industry_map()
    if m:
        picked = {s: m.get(s, "官方無分類") for s in stock_ids}
        return picked, "TWSE OpenAPI t187ap03_L（官方，L21 凍結先驗）"
    if csv_path and csv_path.exists():
        m2 = load_industry_csv(csv_path)
        return ({s: m2.get(s, "CSV無分類") for s in stock_ids},
                f"manual_csv:{csv_path}")
    raise RuntimeError(
        "產業分類：官方 t187ap03_L 不可用且無 industry_map.csv。"
        "請提供 CSV（欄位 stock_id,industry,source，來源需註明），"
        "禁止手寫對照表（L21）。")
