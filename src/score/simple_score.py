# -*- coding: utf-8 -*-
"""簡單評分模型（v3.27）——六因子透明加權，無 ML、無黑箱。

定位（誠實聲明）：**決策輔助檢核分數，非經統計驗證之預測器**。
本專案封版結論為「39 特徵四維度無可重現預測訊號」，本分數不推翻該
結論、不宣稱預測力；它把官方資料整理成一致的檢核視角。

權重＝聖杯（費波那契/黃金）比例，門檻集中於 config/score_config.py。
缺資料 → 權重重分配，不以預設值填補。13MV 下彎 → 總分硬性封頂。
"""
from __future__ import annotations

import pandas as pd

from config.score_config import FACTOR_LABEL, ScoreConfig


def _c01(x) -> float:
    return max(0.0, min(1.0, float(x)))


def _avg(parts: list[tuple[float, str]]) -> tuple[float | None, str]:
    """子項平均（自動略過缺項）；全缺回 None。"""
    ok = [(v, t) for v, t in parts if v is not None]
    if not ok:
        return None, "子項全缺 → 權重重分配"
    return (sum(v for v, _ in ok) / len(ok),
            "；".join(t for _, t in ok))


# ── 因子一：成長動能（營收 YoY + EPS 加速度）──
def score_growth(rev_yoy=None, eps_accel=None,
                 cfg: ScoreConfig = None) -> tuple[float | None, str]:
    cfg = cfg or ScoreConfig()
    parts = []
    if rev_yoy is not None and not pd.isna(rev_yoy):
        v = _c01(rev_yoy / cfg.rev_yoy_full)
        parts.append((v, f"月營收YoY {rev_yoy:.1%}→{v:.2f}"))
    if eps_accel is not None and not pd.isna(eps_accel):
        v = _c01(eps_accel / cfg.eps_accel_full * 0.5 + 0.5)   # 0→0.5
        parts.append((v, f"EPS加速度 {eps_accel:+.1%}→{v:.2f}"))
    return _avg(parts)


# ── 因子二：獲利品質（三率改善 + ROE + FCF）──
def score_quality(margins: dict = None, roe=None, fcf=None,
                  cfg: ScoreConfig = None) -> tuple[float | None, str]:
    cfg = cfg or ScoreConfig()
    parts = []
    if margins:
        imps = [margins.get(k) for k in ("gross_yoy", "op_yoy", "net_yoy")
                if margins.get(k) is not None and not pd.isna(margins.get(k))]
        if imps:
            avg_imp = sum(imps) / len(imps)
            v = _c01(avg_imp / cfg.margin_improve_full * 0.5 + 0.5)
            parts.append((v, f"三率年增 {avg_imp:+.2%}(平均)→{v:.2f}"))
    if roe is not None and not pd.isna(roe):
        v = _c01((roe - cfg.roe_floor) / (cfg.roe_full - cfg.roe_floor))
        parts.append((v, f"ROE {roe:.1%}→{v:.2f}"))
    if fcf is not None and not pd.isna(fcf):
        v = 1.0 if fcf > 0 else 0.0
        parts.append((v, f"自由現金流 {'正' if fcf > 0 else '負'}→{v:.2f}"))
    return _avg(parts)


# ── 因子三：法人動能 ──
def score_institution(inst_net=None, volume=None,
                      cfg: ScoreConfig = None) -> tuple[float | None, str]:
    cfg = cfg or ScoreConfig()
    if inst_net is None or len(pd.Series(inst_net).dropna()) < 5:
        return None, "三大法人資料不足(<5日) → 權重重分配"
    s = pd.Series(inst_net).dropna()
    net5 = float(s.tail(5).sum())
    streak = 0
    for x in reversed(list(s)):
        if x > 0:
            streak += 1
        else:
            break
    p_streak = _c01(streak / cfg.inst_streak_full)
    if volume is not None and len(pd.Series(volume).dropna()) >= 5 and \
            float(pd.Series(volume).tail(5).sum()) > 0:
        ratio = net5 / float(pd.Series(volume).tail(5).sum())
        p_ratio = _c01(ratio / cfg.inst_ratio_full)
        txt = f"近5日淨買超佔量 {ratio:.2%}"
    else:
        p_ratio = 1.0 if net5 > 0 else 0.0
        txt = "成交量缺值，改以淨買超正負計"
    v = 0.5 * p_ratio + 0.5 * p_streak
    return v, f"{txt}；連買 {streak} 日；0.5×{p_ratio:.2f}+0.5×{p_streak:.2f}={v:.2f}"


# ── 因子四：籌碼結構 ──
def score_margin(margin_bal=None, short_bal=None,
                 cfg: ScoreConfig = None) -> tuple[float | None, str]:
    cfg = cfg or ScoreConfig()
    if margin_bal is None or len(pd.Series(margin_bal).dropna()) < 5:
        return None, "融資融券資料不足(<5日) → 權重重分配"
    m = pd.Series(margin_bal).dropna()
    chg = (float(m.iloc[-1]) - float(m.iloc[0])) / max(float(m.iloc[0]), 1.0)
    p_m = _c01(-chg / cfg.margin_drop_full * 0.5 + 0.5)
    if short_bal is not None and len(pd.Series(short_bal).dropna()) >= 1 \
            and float(m.iloc[-1]) > 0:
        sr = float(pd.Series(short_bal).dropna().iloc[-1]) / float(m.iloc[-1])
        p_s = _c01(sr / cfg.short_ratio_full)
        txt = f"券資比 {sr:.1%}"
    else:
        p_s, txt = 0.5, "券資比缺值(中性)"
    v = 0.5 * p_m + 0.5 * p_s
    return v, f"融資餘額 {chg:+.1%}；{txt}；0.5×{p_m:.2f}+0.5×{p_s:.2f}={v:.2f}"


# ── 因子五：量價動能（13MV 三結論）──
def score_tide(mv_short_dir=0, mv_mid_dir=0, veto=False) -> tuple[float, str]:
    if veto or mv_mid_dir < 0:
        return 0.0, "13MV 下彎（絕對否決）→0"
    if mv_short_dir > 0 and mv_mid_dir > 0:
        return 1.0, "5MV+13MV 同步上揚（真波段）→1.0"
    if mv_short_dir > 0:
        return 0.5, "僅 5MV 上揚（短線）→0.5"
    return 0.25, "無攻擊訊號（觀望）→0.25"


# ── 因子六：估值位階 ──
def score_valuation(pe=None, pe_hist=None,
                    cfg: ScoreConfig = None) -> tuple[float | None, str]:
    cfg = cfg or ScoreConfig()
    if pe is None or pd.isna(pe) or pe <= 0:
        return None, "本益比缺值/非正 → 權重重分配"
    if pe_hist is None or len(pd.Series(pe_hist).dropna()) < cfg.min_pe_hist:
        return None, f"本益比歷史不足(<{cfg.min_pe_hist}) → 權重重分配"
    pct = float((pd.Series(pe_hist).dropna() < pe).mean())
    v = 1.0 - pct
    return v, f"PE {pe:.1f} 位於自身第 {pct:.0%} 百分位→{v:.2f}（越便宜越高）"


def compute_simple_score(*, rev_yoy=None, eps_accel=None, margins=None,
                         roe=None, fcf=None, inst_net=None, volume=None,
                         margin_bal=None, short_bal=None, pe=None,
                         pe_hist=None, mv_short_dir=0, mv_mid_dir=0,
                         veto=False, cfg: ScoreConfig = None) -> dict:
    """六因子加權總分（0–100）。缺項權重重分配；13MV 下彎硬性封頂。"""
    cfg = cfg or ScoreConfig()
    parts = {
        "growth": score_growth(rev_yoy, eps_accel, cfg),
        "quality": score_quality(margins, roe, fcf, cfg),
        "institution": score_institution(inst_net, volume, cfg),
        "margin": score_margin(margin_bal, short_bal, cfg),
        "tide": score_tide(mv_short_dir, mv_mid_dir, veto),
        "valuation": score_valuation(pe, pe_hist, cfg),
    }
    avail = {k: v for k, (v, _) in parts.items() if v is not None}
    if not avail:
        return {"score": None, "detail": {}, "n_available": 0}
    w_sum = sum(cfg.weights[k] for k in avail)
    score = sum(cfg.weights[k] * avail[k] for k in avail) / w_sum * 100.0
    capped = bool(veto or mv_mid_dir < 0)
    if capped:
        score = min(score, cfg.veto_score_cap)
    return {
        "score": round(score, 1),
        "detail": {k: {"label": FACTOR_LABEL[k], "value": v,
                       "weight": cfg.weights[k], "calc_logic": t}
                   for k, (v, t) in parts.items()},
        "n_available": len(avail),
        "weight_used": {k: cfg.weights[k] for k in avail},
        "capped_by_13mv": capped,
        "disclaimer": ("決策輔助檢核分數，非經統計驗證之預測器；"
                       "本系統研究結論為四維度 39 特徵無可重現預測訊號。"),
    }
