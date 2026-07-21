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


def summarize_accuracy(ev: pd.DataFrame, n_days: int) -> dict:
    """到期樣本彙總：命中率、二項檢定 p、獨立樣本數、預宣告裁決文字。"""
    from scipy.stats import binomtest
    from src.signal_events import select_independent_dates
    m = ev[ev["matured"] == True]                          # noqa: E712
    n = int(len(m))
    if n == 0:
        return {"n_matured": 0, "n_independent": 0, "hit_rate": None,
                "p_binom": None,
                "verdict": "尚無已到期樣本（每筆需等 5 個交易日後才可評）。"}
    hits = int(m["hit"].sum())
    hr = hits / n
    p = binomtest(hits, n, 0.5).pvalue
    n_ind = 0
    for _sid, g in m.groupby("stock_id"):
        n_ind += len(select_independent_dates(
            sorted(pd.to_datetime(g["last_bar_date"])), n_days))
    if n_ind < 30:
        verdict = (f"累積中（獨立 {n_ind}/30）——依預先宣告規則，"
                   f"達 30 前不得下結論；下方數字僅供追蹤。")
    else:
        sig = p < 0.05
        verdict = (f"獨立樣本已達 {n_ind}（≥30）：命中率 {hr:.1%}、"
                   f"二項檢定 p={p:.3f} → "
                   + ("顯著異於 50%（罕見結果，須複核後才可宣告）"
                      if sig else
                      "與 50% 無顯著差異——前瞻確認模型無判別力，與 "
                      "holdout AUC≈0.494 一致，Model v1 依預宣告規則歸檔。"))
    return {"n_matured": n, "n_independent": int(n_ind),
            "hit_rate": hr, "p_binom": float(p), "verdict": verdict}
