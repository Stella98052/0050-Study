# -*- coding: utf-8 -*-
"""潮汐狀態機（P1，v3.14）：13MV 三結論編碼為單一狀態標籤。

方法論依據（POINT 005 + 13MV 三結論）：
    僅 5MV↑ 而 13MV 未跟上 → 短線，勿輕易抱單
    5MV↑ 且 13MV↑         → 真波段，可持有
    13MV↓                  → 九成波段結束，立即出場（絕對否決，凌駕一切）
    其餘                    → 無攻擊訊號，觀望

身分聲明：方法論檢核值，非經統計驗證之預測訊號（單特徵稽核未過 Holm）。
"""
from __future__ import annotations


def tidal_state(mv_short_dir: int, mv_mid_dir: int,
                veto_active: bool) -> dict:
    """(5MV方向, 13MV方向, 13MV否決) → 狀態標籤。純函式可測。"""
    if veto_active or mv_mid_dir < 0:
        return {"emoji": "🔴", "label": "波段結束",
                "desc": "13MV 下彎＝絕對否決，立即出場不猶豫"}
    if mv_short_dir > 0 and mv_mid_dir > 0:
        return {"emoji": "🟢", "label": "真波段",
                "desc": "5MV+13MV 同步上揚，可持有"}
    if mv_short_dir > 0:
        return {"emoji": "🟡", "label": "僅短線",
                "desc": "5MV 上揚但 13MV 未跟上，勿輕易抱單"}
    return {"emoji": "⚪", "label": "觀望",
            "desc": "無攻擊訊號"}


DIR_TEXT = {1: "↑ 上揚", -1: "↓ 下彎", 0: "→ 持平"}
TIDAL_DISCLAIMER = "潮汐快照為方法論檢核值（官方量價計算），非經統計驗證之預測訊號。"
