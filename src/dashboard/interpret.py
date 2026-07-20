# -*- coding: utf-8 -*-
"""預測卡解讀結論（v3.19）：規則驅動的判讀語句生成。

零猜想原則：每句結論由固定規則機械對應方法論條文（POINT 003/005、
13MV 三結論、VG-6 定案），非自由文字生成。身分：方法論檢核之彙整
陳述，非投資建議、非經統計驗證之預測訊號。
"""
from __future__ import annotations

from src.dashboard.tidal import tidal_state

BURST_TH_NOTE = "（>20%＝波段攻擊量門檻，POINT 003）"


def interpret_card(*, proba: float | None, mv_short_dir: int,
                   mv_mid_dir: int, veto: bool, bias: float | None,
                   burst: bool, wave_label: str | None,
                   is_custom: bool = False) -> dict:
    """本股當前狀態 → 解讀結論三段（方法論判讀/模型定位/行動語意）。"""
    stt = tidal_state(mv_short_dir, mv_mid_dir, veto)
    e = stt["emoji"]

    # ── 段1：方法論判讀（含水位×方向互補解讀）──
    if e == "🔴":
        method = ("13MV 下彎＝波段結束的絕對否決訊號（三結論之三），"
                  "凌駕任何波浪判讀與量能水位。")
    elif e == "🟢":
        method = ("5MV 與 13MV 同步上揚＝真波段（三結論之二），"
                  "量價結構處於攻擊狀態。")
    elif e == "🟡":
        method = ("僅 5MV 上揚、13MV 未跟上＝短線行情（三結論之一），"
                  "勿輕易抱單。")
    else:
        method = "5MV 未上揚＝無攻擊訊號，觀望。"
    if burst and e in ("⚪", "🔴", "🟡"):
        method += (f" 量能乖離 {bias:.1%}🔥為高「水位」{BURST_TH_NOTE}，"
                   "但潮汐狀態顯示「方向」已非上攻——前期爆量餘溫仍在、"
                   "每日量能退潮中（POINT 005：方向重於數值），"
                   "屬「5MV 落則價不易創高」的警戒語境，非進場訊號。")
    elif burst and e == "🟢":
        method += (f" 量能乖離 {bias:.1%}🔥{BURST_TH_NOTE}：攻擊量與"
                   "方向同時成立，波段成色較足。")
    if wave_label not in (None, "", "unknown"):
        method += f" realtime 波浪位置：{wave_label}（規則化近似，標註輔助）。"

    # ── 段2：模型數字定位（誠實固定語句）──
    if is_custom or proba is None:
        model_note = "模型預測不適用此股（不在訓練池），且模型已定案無判別力。"
    else:
        lean = "看多" if proba > 0.5 else "看空/觀望"
        model_note = (f"模型 P(漲)={proba:.1%}（機械標籤：{lean}）——"
                      "VG-6 已定案模型 AUC≈0.5 無判別力，此數字與 50% 的"
                      "差距屬雜訊，判讀時應忽略，僅供前瞻管線演示。")

    # ── 段3：行動語意（方法論條文對應）──
    action = {
        "🔴": "依三結論：立即出場，不可猶豫。",
        "🟢": "依三結論：可持有；一旦 13MV 轉下彎即出場。",
        "🟡": "依三結論：僅短線看待，勿轉為波段持有。",
        "⚪": "等 5MV 方向表態（翻揚且 13MV 跟上才轉攻擊語境）。",
    }[e]

    return {"emoji": e, "state": stt["label"], "method": method,
            "model_note": model_note, "action": action,
            "disclaimer": ("以上為方法論檢核之規則化彙整，非投資建議、"
                           "非經統計驗證之預測訊號。")}
