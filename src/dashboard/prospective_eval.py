# -*- coding: utf-8 -*-
"""前瞻預測準確度檢驗（v3.21）——7/30 Model v1 裁決同一套工具。

原則：
- 只評「已到期」樣本（last_bar_date 之後已有 ≥forward_days 根官方K）
- 淨報酬定義與訓練管線一致：close[t+N]/close[t] − 1 − 總成本率
  （一階近似，成本=買賣手續費+證交稅，Config 寫死來源）
- 獨立樣本計數沿用 prospective_progress 同一規則（逐檔間隔≥N）
- 預先宣告規則：獨立樣本 <30 顯示「累積中，不得下結論」；≥30 依
  二項檢定（vs 50%）與命中率作一次性裁決陳述
"""
from __future__ import annotations

import pandas as pd


def evaluate_from_frames(pred_df: pd.DataFrame,
                         frames: dict[str, pd.DataFrame],
                         cost_total: float, forward_days: int) -> pd.DataFrame:
    """凍結預測 × 官方K → 逐列已到期評估。

    回傳含 realized_net / realized_up / hit / matured 欄；未到期列
    matured=False 且 realized 欄為 NaN（誠實：不足 N 根不評）。"""
    rows = []
    for _, r in pred_df.iterrows():
        sid = str(r["stock_id"])
        d0 = pd.Timestamp(r["last_bar_date"])
        out = {"stock_id": sid, "last_bar_date": d0,
               "proba_up": float(r["proba_up"]), "pick": bool(r["pick"]),
               "matured": False, "realized_net": float("nan"),
               "realized_up": pd.NA, "hit": pd.NA}
        df = frames.get(sid)
        if df is not None and len(df):
            px = df.sort_values("date").reset_index(drop=True)
            px["date"] = pd.to_datetime(px["date"])
            idx = px.index[px["date"] == d0]
            if len(idx) and idx[0] + forward_days < len(px):
                c0 = float(px.loc[idx[0], "close"])
                c1 = float(px.loc[idx[0] + forward_days, "close"])
                net = c1 / c0 - 1.0 - cost_total
                up = net > 0
                out.update({"matured": True, "realized_net": net,
                            "realized_up": up,
                            "hit": bool(r["pick"]) == up})
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_accuracy(ev: pd.DataFrame, n_days: int = 5) -> dict:
    """實績對比（v3.28 簡化）——不用推論統計，直接比對兩個命中率。

    設計理由（使用者指正，2026/7/24）：原以二項檢定對 0.5 求 p 值有二誤：
    ①「淨報酬為正」的自然發生率不是 50%（多頭期間可達 60%），故永遠
    喊多的零技能模型會被判顯著；②同日多檔預測高度相關，獨立性前提
    不成立。正解是直接呈現可驗證的事實：

        模型命中率  vs  永遠喊「看多」的命中率（＝實際上漲比率）

    模型若低於或等於後者，代表它不如一條固定規則——一眼可辨，
    不需要 p 值。
    """
    m = ev[ev["matured"] == True]                          # noqa: E712
    n = int(len(m))
    if n == 0:
        return {"n_matured": 0, "hit_rate": None, "base_rate": None,
                "edge": None, "n_up": 0,
                "verdict": "尚無已到期樣本（每筆需等 5 個交易日後才可評）。"}
    hits = int(m["hit"].sum())
    n_up = int(m["realized_up"].sum())
    hit_rate = hits / n
    base_rate = n_up / n                       # 永遠看多的命中率
    edge = hit_rate - base_rate
    if edge > 0.05:
        verdict = (f"模型命中 {hit_rate:.0%}，高於「永遠看多」基準 "
                   f"{base_rate:.0%}（+{edge:.0%}）——樣本 {n} 筆，"
                   f"持續累積觀察是否維持。")
    elif edge < -0.05:
        verdict = (f"模型命中 {hit_rate:.0%}，**低於**「永遠看多」基準 "
                   f"{base_rate:.0%}（{edge:.0%}）——即模型不如一條固定規則，"
                   f"與 VG-6 判定的無判別力一致。")
    else:
        verdict = (f"模型命中 {hit_rate:.0%}，與「永遠看多」基準 "
                   f"{base_rate:.0%} 相當——無可辨識的優勢。")
    return {"n_matured": n, "hit_rate": hit_rate, "base_rate": base_rate,
            "edge": edge, "n_up": n_up, "verdict": verdict}
