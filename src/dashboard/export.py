# -*- coding: utf-8 -*-
"""資料匯出（v3.10：面板下載按鈕用）。utf-8-sig BOM 讓 Excel 直接開啟不亂碼。"""
from __future__ import annotations

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame → CSV bytes（含 BOM，Excel 相容）。"""
    return df.to_csv(index=False).encode("utf-8-sig")


def coverage_caption(df: pd.DataFrame) -> str:
    """資料涵蓋範圍說明（讓使用者知道歷史資料已自動收集完整）。"""
    if len(df) == 0:
        return "無資料"
    d0 = pd.to_datetime(df["date"].iloc[0]).date()
    d1 = pd.to_datetime(df["date"].iloc[-1]).date()
    return (f"歷史資料已自動收集：{d0} ~ {d1}，共 {len(df)} 根日K"
            f"（官方 TWSE 逐月快取，之後載入免重抓）")
