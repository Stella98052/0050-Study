# -*- coding: utf-8 -*-
"""簡單評分模型（v3.25）——透明加權，無 ML、無黑箱。

定位（誠實聲明）：**決策輔助檢核分數，非經統計驗證之預測器**。
本專案已封版之結論為「39 特徵四維度無可重現預測訊號」，本分數
不推翻該結論、亦不宣稱預測力；它把官方資料整理成一致的檢核視角。

設計原則：
- 每一分項都能說出算法（calc_logic 隨輸出附帶）
- 缺資料 → **權重重分配**，不以預設值填補（不製造假資訊）
- 全部輸入皆官方來源，除股票代號外無人工輸入值
"""
from __future__ import annotations

import pandas as pd

WEIGHTS = {"revenue": 25.0, "valuation": 20.0, "institution": 30.0,
           "margin": 15.0, "tide": 10.0}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_revenue(rev_yoy: float | None) -> tuple[float | None, str]:
    """營收動能：月營收 YoY。0%→0 分、40% 以上→滿分（線性）。"""
    if rev_yoy is None or pd.isna(rev_yoy):
        return None, "月營收 YoY 缺值 → 權重重分配"
    v = _clip01(rev_yoy / 0.40)
    return v, f"rev_yoy={rev_yoy:.1%}；標準化 min(1, YoY/40%)={v:.2f}"


def score_valuation(pe: float | None, pe_hist: pd.Series | None
                    ) -> tuple[float | None, str]:
    """估值位階：本益比在自身近三年分布的百分位，越低分越高。"""
    if pe is None or pd.isna(pe) or pe <= 0:
        return None, "本益比缺值/非正 → 權重重分配"
    if pe_hist is None or len(pe_hist.dropna()) < 60:
        return None, "本益比歷史樣本不足（<60）→ 權重重分配"
    pct = float((pe_hist.dropna() < pe).mean())
    v = 1.0 - pct
    return v, (f"PE={pe:.1f}，位於自身歷史第 {pct:.0%} 百分位；"
               f"分數=1−百分位={v:.2f}（越便宜越高）")


def score_institution(inst_net: pd.Series | None,
                      volume: pd.Series | None) -> tuple[float | None, str]:
    """法人動能：近 5 日淨買超佔成交量比（50% 權重）
    ＋連續買超天數（50% 權重，5 日為滿分）。"""
    if inst_net is None or len(inst_net.dropna()) < 5:
        return None, "三大法人資料不足（<5 日）→ 權重重分配"
    net5 = float(inst_net.tail(5).sum())
    streak = 0
    for x in reversed(list(inst_net.dropna())):
        if x > 0:
            streak += 1
        else:
            break
    part_streak = _clip01(streak / 5.0)
    if volume is not None and len(volume.dropna()) >= 5 and \
            float(volume.tail(5).sum()) > 0:
        ratio = net5 / float(volume.tail(5).sum())
        part_ratio = _clip01(ratio / 0.10)                 # 佔量 10% 即滿分
        txt = f"近5日法人淨買超佔成交量 {ratio:.2%}"
    else:
        part_ratio = 1.0 if net5 > 0 else 0.0
        txt = "成交量缺值，改以淨買超正負計"
    v = 0.5 * part_ratio + 0.5 * part_streak
    return v, (f"{txt}；連續買超 {streak} 日；"
               f"分數=0.5×{part_ratio:.2f}+0.5×{part_streak:.2f}={v:.2f}")


def score_margin(margin_bal: pd.Series | None,
                 short_bal: pd.Series | None) -> tuple[float | None, str]:
    """籌碼結構：融資餘額下降（散戶退場）＋券資比上升（潛在軋空）各半。"""
    if margin_bal is None or len(margin_bal.dropna()) < 5:
        return None, "融資融券資料不足（<5 日）→ 權重重分配"
    m = margin_bal.dropna()
    chg = (float(m.iloc[-1]) - float(m.iloc[0])) / max(float(m.iloc[0]), 1.0)
    part_m = _clip01(-chg / 0.10 * 0.5 + 0.5)              # −10%→1、+10%→0
    if short_bal is not None and len(short_bal.dropna()) >= 1 and \
            float(m.iloc[-1]) > 0:
        sr = float(short_bal.dropna().iloc[-1]) / float(m.iloc[-1])
        part_s = _clip01(sr / 0.30)                        # 券資比 30% 滿分
        txt = f"券資比 {sr:.1%}"
    else:
        part_s, txt = 0.5, "券資比缺值（中性 0.5）"
    v = 0.5 * part_m + 0.5 * part_s
    return v, (f"融資餘額變化 {chg:+.1%}；{txt}；"
               f"分數=0.5×{part_m:.2f}+0.5×{part_s:.2f}={v:.2f}")


def score_tide(mv_short_dir: int, mv_mid_dir: int,
               veto: bool) -> tuple[float | None, str]:
    """量價動能：13MV 三結論。否決→0、真波段→1、僅短線→0.5、觀望→0.25。"""
    if veto or mv_mid_dir < 0:
        return 0.0, "13MV 下彎（絕對否決）→ 0"
    if mv_short_dir > 0 and mv_mid_dir > 0:
        return 1.0, "5MV+13MV 同步上揚（真波段）→ 1.0"
    if mv_short_dir > 0:
        return 0.5, "僅 5MV 上揚（短線）→ 0.5"
    return 0.25, "無攻擊訊號（觀望）→ 0.25"


def compute_simple_score(*, rev_yoy=None, pe=None, pe_hist=None,
                         inst_net=None, volume=None, margin_bal=None,
                         short_bal=None, mv_short_dir=0, mv_mid_dir=0,
                         veto=False) -> dict:
    """五分項加權總分（0–100）。缺項權重重分配；13MV 否決硬性封頂 20。"""
    parts = {
        "revenue": score_revenue(rev_yoy),
        "valuation": score_valuation(pe, pe_hist),
        "institution": score_institution(inst_net, volume),
        "margin": score_margin(margin_bal, short_bal),
        "tide": score_tide(mv_short_dir, mv_mid_dir, veto),
    }
    avail = {k: v for k, (v, _) in parts.items() if v is not None}
    if not avail:
        return {"score": None, "detail": parts, "n_available": 0,
                "note": "無任何分項可用"}
    w_sum = sum(WEIGHTS[k] for k in avail)
    score = sum(WEIGHTS[k] * avail[k] for k in avail) / w_sum * 100.0
    capped = False
    if veto or mv_mid_dir < 0:
        # 方法論鐵律優先於任何加總：13MV 下彎不得呈現高分
        score = min(score, 20.0)
        capped = True
    return {
        "score": round(score, 1),
        "detail": {k: {"value": v, "calc_logic": t} for k, (v, t) in parts.items()},
        "n_available": len(avail),
        "weight_used": {k: WEIGHTS[k] for k in avail},
        "capped_by_13mv": capped,
        "disclaimer": ("決策輔助檢核分數，非經統計驗證之預測器；"
                       "本系統研究結論為四維度 39 特徵無可重現預測訊號。"),
    }
