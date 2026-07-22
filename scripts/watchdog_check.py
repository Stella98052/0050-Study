# -*- coding: utf-8 -*-
"""watchdog 資料驅動檢查（v3.23）：純標準庫，無需安裝依賴。

判定邏輯（取代「比對日曆今日」的舊法）：
    stale  ⇔  predictions 最新資料日 < TWSE 最新交易日
以官方實際有的交易日為基準，故休市日/颱風日/連假一律不誤報，
亦不受排程延遲影響。TWSE 不可達時回報並放棄判定（不誤觸發）。
"""
from __future__ import annotations

import csv
import datetime
import json
import urllib.request
from pathlib import Path

_URL = ("https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        "?response=json&date={ym}01&stockNo={sid}")
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.twse.com.tw/",
}


def roc_to_iso(roc: str) -> str:
    """民國日期（'115/07/22' 或 '115年07月22日'）→ ISO 西元字串。"""
    t = str(roc).strip().replace("年", "/").replace("月", "/").rstrip("日")
    y, m, d = t.split("/")
    return datetime.date(int(y) + 1911, int(m), int(d)).isoformat()


def parse_stock_day_dates(payload: dict) -> list[str]:
    """STOCK_DAY JSON → 交易日 ISO 清單（stat 非 OK 或無資料回空）。"""
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        return []
    out = []
    for row in payload.get("data", []) or []:
        try:
            out.append(roc_to_iso(row[0]))
        except Exception:                                  # noqa: BLE001
            continue
    return sorted(out)


def predictions_latest(path: Path) -> str | None:
    """predictions.csv 最新資料日（無檔或空回 None）。"""
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        ds = [r.get("last_bar_date", "") for r in csv.DictReader(f)]
    ds = [d for d in ds if d]
    return max(ds) if ds else None


def decide_stale(pred_latest: str | None,
                 twse_latest: str | None) -> bool | None:
    """None＝無法判定（TWSE 不可達）；True＝落後需補跑。"""
    if twse_latest is None:
        return None
    if pred_latest is None:
        return True
    return pred_latest < twse_latest


def twse_latest_trading_day(sid: str = "2330", timeout: int = 20) -> str | None:
    """官方最新交易日（本月無資料時退看上個月，處理月初情形）。"""
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)  # 台北
    for back in (0, 1):
        y, m = now.year, now.month - back
        if m <= 0:
            y, m = y - 1, m + 12
        url = _URL.format(ym=f"{y}{m:02d}", sid=sid)
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dates = parse_stock_day_dates(json.load(resp))
        except Exception as exc:                           # noqa: BLE001
            print(f"  [watchdog] TWSE 讀取失敗：{type(exc).__name__}: {exc}")
            return None
        if dates:
            return dates[-1]
    return None


def main() -> int:
    import os
    pred = predictions_latest(Path("data/predictions.csv"))
    twse = twse_latest_trading_day()
    stale = decide_stale(pred, twse)
    print(f"predictions 最新={pred}｜TWSE 最新交易日={twse}｜stale={stale}")
    if stale is None:
        print("  無法判定（官方不可達）——本次不觸發、不開 Issue")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as g:
            g.write(f"stale={'true' if stale else 'false'}\n")
            g.write(f"pred_latest={pred}\n")
            g.write(f"twse_latest={twse}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
