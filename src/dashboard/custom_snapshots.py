# -*- coding: utf-8 -*-
"""自選股每日技術快照紀錄（v3.22）。

由每日排程對 repo 內 custom_watchlist.csv 的股票累積：收盤、5/13MV
方向、潮汐狀態、量能乖離、波浪位置——全為方法論檢核值，不含模型
數字（模型對自選股無效）。同（代號,資料日）去重；面板讀取顯示歷史。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAP_COLS = ["run_ts", "stock_id", "last_bar_date", "close",
             "mv_short_dir", "mv_mid_dir", "veto", "bias", "burst",
             "tidal", "wave"]


def build_snapshot_row(stock_id: str, ohlcv_last: pd.Series,
                       feats_last: pd.Series, run_ts: str) -> dict:
    """K線末列（close/date）＋特徵末列（MV/波浪）→ 快照列（純函式）。
    註：特徵矩陣不含 close 欄，價格必須取自 ohlcv（L53）。"""
    from src.dashboard.tidal import tidal_state
    sd = int(feats_last.get("mv_short_direction", 0))
    md = int(feats_last.get("mv_mid_direction", 0))
    veto = bool(feats_last.get("mv_mid_veto_active", False))
    stt = tidal_state(sd, md, veto)
    bias = feats_last.get("mv_bias")
    return {"run_ts": run_ts, "stock_id": str(stock_id),
            "last_bar_date": str(pd.Timestamp(ohlcv_last["date"]).date()),
            "close": float(ohlcv_last["close"]),
            "mv_short_dir": sd, "mv_mid_dir": md, "veto": veto,
            "bias": float(bias) if pd.notna(bias) else float("nan"),
            "burst": bool(feats_last.get("is_volume_burst", False)),
            "tidal": f"{stt['emoji']} {stt['label']}",
            "wave": str(feats_last.get("wave_label_realtime", ""))}


def append_snapshot(row: dict, path: Path) -> bool:
    """附加快照（同代號+資料日已存在則略過）。回傳是否寫入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path, dtype={"stock_id": str})
        dup = ((df["stock_id"] == row["stock_id"]) &
               (df["last_bar_date"] == row["last_bar_date"])).any()
        if dup:
            return False
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=SNAP_COLS)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def load_snapshots(path: Path, stock_id: str | None = None) -> pd.DataFrame:
    """讀取快照歷史（全部或單股，依資料日排序）。"""
    if not Path(path).exists():
        return pd.DataFrame(columns=SNAP_COLS)
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["last_bar_date"] = pd.to_datetime(df["last_bar_date"])
    if stock_id:
        df = df[df["stock_id"] == str(stock_id)]
    return df.sort_values("last_bar_date").reset_index(drop=True)
