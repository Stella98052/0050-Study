# -*- coding: utf-8 -*-
"""自選股資料儲存（v3.16 收官項）：與前十大同管線抓取、產物隔離存放。

治理三則：
①同軌——抓取/特徵與模型池走完全相同引擎（fetch_stock_history/
  build_feature_matrix），無第二套邏輯
②隔離——產物寫 data/custom/，不碰 holdings/predictions/模型池
③防誤用——特徵匯出剔除未來報酬欄（與面板下載一致）
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_FUTURE_COLS = ("fwd_return_gross", "fwd_return_net", "label_up")


def save_custom_stock(stock_id: str, p1, p2, outdir: Path,
                      fetch_fn=None, with_valuation: bool = False) -> dict:
    """抓取單一自選股十年資料並存檔。回傳摘要 dict（供 CLI 列印）。"""
    from src.features.feature_matrix import build_feature_matrix
    from src.fetch.twse_daily import fetch_stock_history
    fetch = fetch_fn or fetch_stock_history
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    df = fetch(stock_id, start, end, p1)
    if len(df) == 0:
        return {"stock_id": stock_id, "rows": 0, "ok": False,
                "msg": "查無官方日K（上櫃/興櫃或代號不存在）"}
    df = df.sort_values("date").reset_index(drop=True)
    # 輕量完整性檢查（VG-1 精神：重複/跳空回報，不擋存檔）
    dup = int(df["date"].duplicated().sum())
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stock_id}_ohlcv.csv", index=False,
              encoding="utf-8-sig")
    feats = build_feature_matrix(df, p1, p2)
    feats = feats.drop(columns=[c for c in _FUTURE_COLS if c in feats.columns])
    feats.to_csv(outdir / f"{stock_id}_features.csv", index=False,
                 encoding="utf-8-sig")
    msg_extra = ""
    if with_valuation:
        from src.fetch.twse_bwibbu import fetch_valuation_history
        val = fetch_valuation_history(stock_id, start, end, p1)
        if len(val):
            val.to_csv(outdir / f"{stock_id}_valuation.csv", index=False,
                       encoding="utf-8-sig")
            msg_extra = f"｜估值 {len(val)} 列"
    return {"stock_id": stock_id, "rows": len(df), "ok": True,
            "msg": f"{df['date'].min().date()}~{df['date'].max().date()}"
                   f"｜重複日 {dup}{msg_extra}"}


def load_watchlist(path: Path) -> list[str]:
    """凍結清單檔（stock_id 欄；防偏誤的預先登記機制）。"""
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, comment="#")
    col = "stock_id" if "stock_id" in df.columns else df.columns[0]
    return [c.strip() for c in df[col].dropna() if str(c).strip().isdigit()]


def build_watchlist_zip(stock_ids: list[str], p1, p2,
                        with_valuation: bool = False,
                        fetch_fn=None) -> tuple[bytes, list[dict]]:
    """自選清單 → 記憶體 ZIP（面板一鍵下載用；與 CLI 完全同引擎）。

    回傳 (zip_bytes, 各股摘要)。瀏覽器無法讓伺服器直接寫檔到使用者
    電腦，故面板內儲存的正確形態＝打包後下載。"""
    import io
    import zipfile as _zf
    from datetime import date, timedelta
    from src.features.feature_matrix import build_feature_matrix
    from src.fetch.twse_daily import fetch_stock_history
    fetch = fetch_fn or fetch_stock_history
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    buf = io.BytesIO()
    summaries = []
    with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as z:
        for sid in dict.fromkeys(stock_ids):
            try:
                df = fetch(sid, start, end, p1)
            except Exception as exc:                      # noqa: BLE001
                summaries.append({"stock_id": sid, "ok": False,
                                  "msg": f"{type(exc).__name__}: {exc}"})
                continue
            if len(df) == 0:
                summaries.append({"stock_id": sid, "ok": False,
                                  "msg": "查無官方日K（上櫃/興櫃或代號不存在）"})
                continue
            df = df.sort_values("date").reset_index(drop=True)
            z.writestr(f"{sid}_ohlcv.csv",
                       df.to_csv(index=False).encode("utf-8-sig"))
            feats = build_feature_matrix(df, p1, p2)
            feats = feats.drop(columns=[c for c in _FUTURE_COLS
                                        if c in feats.columns])
            z.writestr(f"{sid}_features.csv",
                       feats.to_csv(index=False).encode("utf-8-sig"))
            extra = ""
            if with_valuation:
                from src.fetch.twse_bwibbu import fetch_valuation_history
                val = fetch_valuation_history(sid, start, end, p1)
                if len(val):
                    z.writestr(f"{sid}_valuation.csv",
                               val.to_csv(index=False).encode("utf-8-sig"))
                    extra = f"｜估值 {len(val)} 列"
            summaries.append({"stock_id": sid, "ok": True,
                              "msg": f"{len(df)} 根日K{extra}"})
    return buf.getvalue(), summaries
