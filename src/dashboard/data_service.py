# -*- coding: utf-8 -*-
"""面板資料服務：單一股票的 K線 + 兩版波浪轉折 + MV 潮汐 + 最新特徵列。
全部委派 phase1/phase2 既有函式（單一事實來源），本模組僅組裝。"""
from __future__ import annotations
from datetime import date, timedelta

import pandas as pd

from config.config import Config
from config.phase2_config import Phase2Config
from config.phase3_config import Phase3Config
from src.fetch.twse_daily import fetch_stock_history
from src.features.feature_matrix import build_feature_matrix
from src.volume.volume_features import compute_mv_features
from src.wave.zigzag import compute_pivots_realtime, compute_pivots_retrospective


def load_stock_view(stock_id: str, p1: Config, p2: Phase2Config,
                    p3: Phase3Config) -> dict:
    """組裝面板所需之單股視圖（抓取走 phase1 快取與禮貌延遲）。"""
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    df = fetch_stock_history(stock_id, start, end, p1)
    if len(df) == 0:
        raise ValueError(
            f"{stock_id} 查無官方日K資料：可能為上櫃(TPEx)/興櫃股票或"
            f"代號不存在——本系統目前僅支援上市(TWSE)股票。")
    df = df.sort_values("date").reset_index(drop=True)
    piv_retro = compute_pivots_retrospective(df, p1)
    piv_rt = compute_pivots_realtime(df, end, p1)
    mv = compute_mv_features(df, p1)
    feats = build_feature_matrix(df, p1, p2)
    tail = df.tail(p3.chart_lookback_days).reset_index(drop=True)
    return {"ohlcv": df, "ohlcv_tail": tail,
            "pivots_retro": piv_retro, "pivots_rt": piv_rt,
            "mv": mv, "features": feats,
            "last_bar_date": str(pd.to_datetime(df["date"].iloc[-1]).date()),
            "last_close": float(df["close"].iloc[-1])}
