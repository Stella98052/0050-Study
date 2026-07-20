# -*- coding: utf-8 -*-
"""TWSE 官方個股日估值（BWIBBU：本益比/殖利率/股價淨值比）。P2 首項。

端點：https://www.twse.com.tw/exchangeReport/BWIBBU?response=json&date=YYYYMM01&stockNo=XXXX
月查詢；欄位（官方 fields 順序）：
    日期(民國) | 殖利率(%) | 股利年度 | 本益比 | 股價淨值比 | 財報年/季
防前視性質：官方以「當日收盤價 ÷ 已公告之最新財報 EPS」計算，
數值於當日收盤後即為已知——天然 point-in-time，無需發布日對齊。
快取/重試/標頭/越月防護全數比照 twse_daily（單一抓法治理）。
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.fetch.twse_daily import (_TWSE_HEADERS, _atomic_to_csv,
                                  _roc_date_to_ad, month_range)


def _roc_any_to_ad(roc: str) -> date:
    """民國日期雙格式解析：'115/06/01' 與 '115年06月01日' 皆可
    （BWIBBU 實抓確認用中文格式，與 STOCK_DAY 斜線格式不同——L40）。"""
    t = str(roc).strip().replace("年", "/").replace("月", "/").rstrip("日")
    return _roc_date_to_ad(t)

_BWIBBU_URL = "https://www.twse.com.tw/exchangeReport/BWIBBU"
VAL_COLS = ["stock_id", "date", "dividend_yield", "pe_ratio", "pb_ratio"]


def _num(x) -> float:
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_bwibbu_json(payload: dict, stock_id: str) -> pd.DataFrame:
    """官方 JSON → DataFrame。

    L41：BWIBBU 欄位數/順序歷史上改版過（舊年代 4 欄「日期,本益比,
    殖利率(%),股價淨值比」；新版 6 欄含股利年度/財報年季）——固定位置
    索引在舊列越界或錯位。改以官方 fields 欄名動態對映，兩代皆正確。"""
    if payload.get("stat") != "OK" or not payload.get("data"):
        return pd.DataFrame(columns=VAL_COLS)
    fields = [str(f) for f in payload.get("fields", [])]

    def _find(*keys):
        for i, f in enumerate(fields):
            if any(k in f for k in keys):
                return i
        return None

    i_dy = _find("殖利率")
    i_pe = _find("本益比")
    i_pb = _find("淨值比")
    rows = []
    for r in payload["data"]:
        rows.append({
            "stock_id": stock_id,
            "date": pd.Timestamp(_roc_any_to_ad(r[0])),
            "dividend_yield": _num(r[i_dy]) if i_dy is not None
            and i_dy < len(r) else float("nan"),
            "pe_ratio": _num(r[i_pe]) if i_pe is not None
            and i_pe < len(r) else float("nan"),
            "pb_ratio": _num(r[i_pb]) if i_pb is not None
            and i_pb < len(r) else float("nan"),
        })
    return pd.DataFrame(rows, columns=VAL_COLS)


def fetch_valuation_history(stock_id: str, start: date, end: date, cfg,
                            session: requests.Session | None = None
                            ) -> pd.DataFrame:
    """逐月抓取估值並快取於 data/raw_valuation/{sid}/YYYYMM.csv。"""
    sess = session or requests.Session()
    cache_root = Path(getattr(cfg, "cache_dir", Path("data/raw"))).parent \
        / "raw_valuation" / stock_id
    cache_root.mkdir(parents=True, exist_ok=True)
    frames = []
    today_ym = f"{date.today().year}{date.today().month:02d}"
    for ym in month_range(start, end):
        cpath = cache_root / f"{ym}.csv"
        if cpath.exists() and ym != today_ym:            # 當月不快取（會增長）
            frames.append(pd.read_csv(cpath, parse_dates=["date"],
                                      dtype={"stock_id": str}))
            continue
        params = {"response": "json", "date": f"{ym}01", "stockNo": stock_id}
        df = pd.DataFrame(columns=VAL_COLS)
        last_err = None
        for attempt in range(3):
            try:
                resp = sess.get(_BWIBBU_URL, params=params,
                                headers=_TWSE_HEADERS,
                                timeout=getattr(cfg, "request_timeout_sec", 15))
                resp.raise_for_status()
                payload = resp.json()
                df = parse_bwibbu_json(payload, stock_id)
                if len(df) == 0:                          # 解析空：印官方回應線索
                    last_err = (f"stat={payload.get('stat')!r} "
                                f"keys={list(payload.keys())[:6]}")
                break
            except Exception as exc:                     # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        if len(df) == 0:
            # L29/L40：失敗必印原因；空結果不快取（避免快取中毒，下次重抓）
            print(f"  [BWIBBU] {stock_id} {ym} 無資料｜{last_err}")
            frames.append(df)
            time.sleep(getattr(cfg, "request_delay_sec", 1.5))
            continue
        # 越月防護：只留該月列（比照 twse_daily L 教訓）
        df = df[df["date"].dt.strftime("%Y%m") == ym]
        _atomic_to_csv(df, cpath)
        frames.append(df)
        time.sleep(getattr(cfg, "request_delay_sec", 1.5))
    out = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=VAL_COLS)
    if len(out) == 0:
        print(f"  [BWIBBU] {stock_id} 全期間無資料——請以診斷指令檢視官方"
              f"原始回應（status/stat/欄位鍵名），勿臆測原因（L29）")
        return out
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    m = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
    return out[m].reset_index(drop=True)


def add_valuation_features(feats: pd.DataFrame,
                           val: pd.DataFrame) -> pd.DataFrame:
    """估值三欄併入特徵矩陣（同日左併；缺日 NaN，不前向填充——
    BWIBBU 為每日資料，缺值即當日官方未提供，不以舊值冒充）。"""
    out = feats.copy()
    out["date"] = pd.to_datetime(out["date"])
    v = val[["date", "dividend_yield", "pe_ratio", "pb_ratio"]].copy()
    v["date"] = pd.to_datetime(v["date"])
    return out.merge(v, on="date", how="left")
