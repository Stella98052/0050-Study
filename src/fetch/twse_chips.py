# -*- coding: utf-8 -*-
"""籌碼面官方資料（v3.25）：三大法人買賣超 T86、融資融券 MI_MARGN。

兩者皆為「單日全市場」端點（與 STOCK_DAY 的「單股整月」相反），
故按日抓取並快取於 data/raw_chips/{T86|MARGIN}/YYYYMMDD.csv。

防前視：兩者皆為當日盤後公布之當日資料，t 日收盤後即已知，無時滯。
解析一律欄名驅動（L41）：官方欄序歷年變動過，不可用固定索引。
失敗必印原因、空結果不快取（L40/L42）。
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.fetch.twse_daily import _TWSE_HEADERS, _atomic_to_csv

# 解析器版本：解析邏輯每次修正即 +1，快取目錄隨之改變，舊快取自動失效。
# （L61：BWIBBU 與 T86 皆曾因「舊解析器寫入的快取」讓修正無效——
#  修好解析器卻讀到舊檔，症狀與未修相同，極難察覺）
PARSER_VERSION = "v2"

_T86_URL = "https://www.twse.com.tw/fund/T86"
_MARGIN_URL = "https://www.twse.com.tw/exchangeReport/MI_MARGN"

INST_COLS = ["stock_id", "date", "inst_net", "trust_net", "foreign_net"]
MARGIN_COLS = ["stock_id", "date", "margin_bal", "short_bal"]


def _norm(c) -> str:
    """欄名正規化：去空白（官方表頭常含全形空白/換行）。"""
    return "".join(str(c).split())


def _num(x) -> float:
    t = str(x).replace(",", "").strip()
    if t in ("", "-", "--", "nan"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _pick(fields: list[str], *keys, exclude=()) -> int | None:
    """依關鍵字找欄位索引（全部 keys 皆須命中、exclude 須不命中）。"""
    for i, f in enumerate(fields):
        if all(k in f for k in keys) and not any(x in f for x in exclude):
            return i
    return None


def parse_t86(payload: dict, d: date) -> pd.DataFrame:
    """T86 JSON → [stock_id, date, inst_net, trust_net, foreign_net]（股數）。

    官方欄名例：證券代號／三大法人買賣超股數／投信買賣超股數／
    外陸資買賣超股數(不含外資自營商)。欄序歷年有變，故欄名驅動。
    """
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=INST_COLS)
    fields = [_norm(f) for f in (payload.get("fields") or [])]
    rows = payload.get("data") or []
    if not fields or not rows:
        return pd.DataFrame(columns=INST_COLS)
    # L58：索引 0 是 falsy，不可用 `or` 串接備援（第一欄會被誤判為缺欄）
    i_id = _pick(fields, "證券代號")
    if i_id is None:
        i_id = _pick(fields, "代號")
    i_all = _pick(fields, "三大法人買賣超")
    i_trust = _pick(fields, "投信", "買賣超")
    # L60：官方欄名為「外陸資買賣超股數(不含外資自營商)」——括號內含
    # 「自營」二字，故不可用 exclude=("自營",)，須以「外陸資」正面指名
    i_fore = _pick(fields, "外陸資", "買賣超")
    if i_fore is None:
        i_fore = _pick(fields, "外資", "買賣超", exclude=("自營商買賣超",))
    if i_id is None or i_all is None:
        return pd.DataFrame(columns=INST_COLS)
    out = []
    for r in rows:
        sid = str(r[i_id]).strip()
        if not sid.isdigit():
            continue
        out.append({
            "stock_id": sid, "date": pd.Timestamp(d),
            "inst_net": _num(r[i_all]),
            "trust_net": _num(r[i_trust]) if i_trust is not None
            and i_trust < len(r) else float("nan"),
            "foreign_net": _num(r[i_fore]) if i_fore is not None
            and i_fore < len(r) else float("nan"),
        })
    return pd.DataFrame(out, columns=INST_COLS)


def parse_margin(payload: dict, d: date) -> pd.DataFrame:
    """MI_MARGN JSON → [stock_id, date, margin_bal, short_bal]（張/千股）。

    個股表可能位於 tables[] 或 data 之一；以「股票代號」欄名定位該表。
    融資今日餘額／融券今日餘額同樣欄名驅動（兩者欄名皆為「今日餘額」，
    以出現順序區分：先融資後融券，為官方固定版面）。
    """
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=MARGIN_COLS)
    cands = []
    if payload.get("tables"):
        cands = [(t.get("fields") or [], t.get("data") or [])
                 for t in payload["tables"]]
    else:
        cands = [(payload.get("fields") or [], payload.get("data") or [])]
    for raw_fields, rows in cands:
        fields = [_norm(f) for f in raw_fields]
        # 實抓確認（2026/7/24）：融資融券彙總表欄名為「代號」，
        # 非「股票代號」／「證券代號」——三者皆列為候選（順序由嚴至寬）
        i_id = _pick(fields, "股票代號")
        if i_id is None:
            i_id = _pick(fields, "證券代號")
        if i_id is None:
            i_id = _pick(fields, "代號")
        if i_id is None or not rows:
            continue
        bal_idx = [i for i, f in enumerate(fields) if "今日餘額" in f]
        if len(bal_idx) < 2:
            continue
        i_m, i_s = bal_idx[0], bal_idx[1]
        out = []
        for r in rows:
            sid = str(r[i_id]).strip()
            if not sid.isdigit():
                continue
            out.append({"stock_id": sid, "date": pd.Timestamp(d),
                        "margin_bal": _num(r[i_m]),
                        "short_bal": _num(r[i_s])})
        if out:
            return pd.DataFrame(out, columns=MARGIN_COLS)
    return pd.DataFrame(columns=MARGIN_COLS)


def describe_payload(payload: dict, max_tables: int = 4) -> str:
    """回應結構摘要（stat=OK 卻解析零列時用於定位，不臆測欄名）。"""
    if not isinstance(payload, dict):
        return f"非 dict：{type(payload).__name__}"
    parts = [f"top_keys={list(payload.keys())[:12]}"]
    if isinstance(payload.get("fields"), list):
        parts.append(f"fields={[str(f) for f in payload['fields']][:14]}")
        parts.append(f"n_data={len(payload.get('data') or [])}")
    tabs = payload.get("tables")
    if isinstance(tabs, list):
        parts.append(f"n_tables={len(tabs)}")
        for i, t in enumerate(tabs[:max_tables]):
            if isinstance(t, dict):
                parts.append(
                    f"  table[{i}] title={str(t.get('title'))[:30]!r} "
                    f"fields={[str(f) for f in (t.get('fields') or [])][:14]} "
                    f"n_data={len(t.get('data') or [])}")
    return "\n    ".join(parts)


def _fetch_day(url: str, params: dict, parser, d: date, kind: str,
               cfg, sess) -> pd.DataFrame:
    """單日抓取＋快取（空結果不快取；失敗必印原因）。"""
    cache = (Path("data/raw_chips") / f"{kind}_{PARSER_VERSION}"
             / f"{d:%Y%m%d}.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)        # L59：先建目錄
    if cache.exists():
        df = pd.read_csv(cache, dtype={"stock_id": str}, parse_dates=["date"])
        return df
    last_err = None
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            resp = sess.get(url, params=params, headers=_TWSE_HEADERS,
                            timeout=getattr(cfg, "request_timeout_sec", 15))
            resp.raise_for_status()
            payload = resp.json()
            df = parser(payload, d)
            if len(df) == 0:
                last_err = f"stat={payload.get('stat')!r}"
                if str(payload.get("stat")) == "OK":
                    # 官方回 OK 卻解析零列＝欄名/結構與假設不符，印結構定位
                    last_err += ("\n    ⚠ stat=OK 但解析零列，實際結構：\n    "
                                 + describe_payload(payload))
            break
        except Exception as exc:                           # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    time.sleep(getattr(cfg, "request_delay_sec", 1.5))
    if len(df) == 0:
        print(f"  [{kind}] {d:%Y-%m-%d} 無資料｜{last_err}（休市日屬正常）")
        return df
    _atomic_to_csv(df, cache)
    return df


def fetch_chips_recent(stock_ids: list[str], end: date, days: int, cfg,
                       session=None) -> pd.DataFrame:
    """近 days 個日曆日的籌碼資料（休市日自動略過）。

    回傳長表 [stock_id, date, inst_net, trust_net, foreign_net,
    margin_bal, short_bal]，僅含 stock_ids。
    """
    sess = session or requests.Session()
    ids = set(map(str, stock_ids))
    inst_parts, marg_parts = [], []
    for k in range(days):
        d = end - timedelta(days=k)
        if d.weekday() >= 5:                               # 週末必無資料
            continue
        ymd = f"{d:%Y%m%d}"
        t86 = _fetch_day(_T86_URL, {"response": "json", "date": ymd,
                                    "selectType": "ALL"},
                         parse_t86, d, "T86", cfg, sess)
        if len(t86):
            inst_parts.append(t86[t86["stock_id"].isin(ids)])
        mg = _fetch_day(_MARGIN_URL, {"response": "json", "date": ymd,
                                      "selectType": "ALL"},
                        parse_margin, d, "MARGIN", cfg, sess)
        if len(mg):
            marg_parts.append(mg[mg["stock_id"].isin(ids)])
    inst = (pd.concat(inst_parts, ignore_index=True) if inst_parts
            else pd.DataFrame(columns=INST_COLS))
    marg = (pd.concat(marg_parts, ignore_index=True) if marg_parts
            else pd.DataFrame(columns=MARGIN_COLS))
    if len(inst) == 0 and len(marg) == 0:
        return pd.DataFrame(columns=INST_COLS + MARGIN_COLS[2:])
    if len(inst) == 0:
        return marg.sort_values(["stock_id", "date"]).reset_index(drop=True)
    if len(marg) == 0:
        return inst.sort_values(["stock_id", "date"]).reset_index(drop=True)
    out = inst.merge(marg, on=["stock_id", "date"], how="outer")
    return out.sort_values(["stock_id", "date"]).reset_index(drop=True)
