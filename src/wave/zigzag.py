# -*- coding: utf-8 -*-
"""ZigZag 轉折點偵測（retrospective / realtime 雙版本）。

演算法（單向前向掃描，兩版本共用同一核心，僅輸出視角不同）：
    1. 峰以 high、谷以 low 追蹤極值。
    2. 自上一極值反向移動幅度 ≥ zigzag_threshold 的「當日」，該極值被確認為
       轉折點，confirmed_date = 當日（realtime 延遲的來源）。
    3. retrospective 版本：回傳全部確認轉折 + 最後一個「暫定極值」，
       confirmed_date 一律 None（僅供視覺化，禁入訓練管線）。
    4. realtime 版本：僅回傳 confirmed_date <= as_of 的轉折；
       掃描只使用 date <= as_of 的資料（防未來函數，pytest 竄改測試把關）。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from config.config import Config
from src.schemas import Pivot


def _scan_pivots(df: pd.DataFrame, threshold: float) -> tuple[list[Pivot], Pivot | None]:
    """前向掃描核心。

    回傳 (已確認轉折清單, 最後暫定極值或 None)。
    確認規則：peak 確認 = low 自波段最高點回落 ≥ threshold；
              trough 確認 = high 自波段最低點回升 ≥ threshold。
    """
    if len(df) == 0:
        return [], None
    dates = pd.to_datetime(df["date"]).dt.date.tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()

    confirmed: list[Pivot] = []
    # direction: 0 未定, +1 追蹤峰（上升段）, -1 追蹤谷（下降段）
    direction = 0
    ext_idx, ext_price = 0, highs[0]          # 目前極值（依 direction 解讀）
    min_idx, min_price = 0, lows[0]           # 未定向階段同時追蹤高低
    max_idx, max_price = 0, highs[0]

    for i in range(1, len(df)):
        if direction == 0:
            if highs[i] > max_price:
                max_idx, max_price = i, highs[i]
            if lows[i] < min_price:
                min_idx, min_price = i, lows[i]
            if lows[i] <= max_price * (1 - threshold):
                confirmed.append(
                    Pivot(max_idx, dates[max_idx], max_price, "peak", dates[i])
                )
                direction, ext_idx, ext_price = -1, i, lows[i]
            elif highs[i] >= min_price * (1 + threshold):
                confirmed.append(
                    Pivot(min_idx, dates[min_idx], min_price, "trough", dates[i])
                )
                direction, ext_idx, ext_price = +1, i, highs[i]
        elif direction == +1:                 # 上升段，追蹤峰
            if highs[i] > ext_price:
                ext_idx, ext_price = i, highs[i]
            elif lows[i] <= ext_price * (1 - threshold):
                confirmed.append(
                    Pivot(ext_idx, dates[ext_idx], ext_price, "peak", dates[i])
                )
                direction, ext_idx, ext_price = -1, i, lows[i]
        else:                                 # 下降段，追蹤谷
            if lows[i] < ext_price:
                ext_idx, ext_price = i, lows[i]
            elif highs[i] >= ext_price * (1 + threshold):
                confirmed.append(
                    Pivot(ext_idx, dates[ext_idx], ext_price, "trough", dates[i])
                )
                direction, ext_idx, ext_price = +1, i, highs[i]

    tentative: Pivot | None = None
    if direction != 0:
        kind = "peak" if direction == +1 else "trough"
        tentative = Pivot(ext_idx, dates[ext_idx], ext_price, kind, None)
    return confirmed, tentative


def compute_pivots_retrospective(df: pd.DataFrame, cfg: Config) -> list[Pivot]:
    """◇ 回溯版：全部轉折 + 最後暫定極值；confirmed_date 一律 None。

    僅供圖表視覺化。禁止流入訓練 / 驗證 / 參數選擇（VG-5 斷言把關）。
    """
    confirmed, tentative = _scan_pivots(df, cfg.zigzag_threshold)
    out = [
        Pivot(p.bar_index, p.pivot_date, p.price, p.kind, None) for p in confirmed
    ]
    if tentative is not None:
        out.append(tentative)
    return out


def compute_pivots_realtime(
    df: pd.DataFrame, as_of: date, cfg: Config
) -> list[Pivot]:
    """★ 即時版：僅使用 date <= as_of 的資料掃描，僅回傳已確認轉折。

    保證：所有回傳 Pivot 之 confirmed_date <= as_of，且 as_of 之後的
    任何資料異動不影響輸出（test_no_lookahead 竄改測試驗證）。
    """
    mask = pd.to_datetime(df["date"]).dt.date <= as_of
    confirmed, _tentative = _scan_pivots(
        df.loc[mask].reset_index(drop=True), cfg.zigzag_threshold
    )
    return confirmed
