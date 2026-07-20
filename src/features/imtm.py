# -*- coding: utf-8 -*-
"""IMTM 產業動能——池內同業版（P2 第三項首版）。

定義（沿 Workstation 方法論）：MTM_N = (今收 − N日前收)/N日前收；
IMTM = 同產業「其他成員」之 MTM 均值（排除自身，防自我相關）。
分組凍結官方分類（t187ap03_L / CSV fallback，L21）。
限制（誠實標註）：本版同業僅及於股池內成員（半導體約4同業），
是「池內同業動能」代理；全產業版待池子母體化後接產業指數。
同業 <1 → NaN。同日橫斷面資訊於收盤即已知，無前視。"""
from __future__ import annotations

import pandas as pd

IMTM_COLS = ["imtm_5d_peer", "imtm_20d_peer"]


def add_peer_momentum(frames: dict[str, pd.DataFrame],
                      industry_of: dict[str, str]) -> dict[str, pd.DataFrame]:
    """{sid: ohlcv} → {sid: DataFrame[date, imtm_5d_peer, imtm_20d_peer]}。"""
    mtm = {}
    for sid, df in frames.items():
        c = df.sort_values("date")["close"].astype("float64")
        mtm[sid] = pd.DataFrame({
            "date": pd.to_datetime(df.sort_values("date")["date"]).values,
            "m5": c.pct_change(5).values,
            "m20": c.pct_change(20).values,
            "sid": sid,
        })
    allm = pd.concat(mtm.values(), ignore_index=True)
    allm["ind"] = allm["sid"].map(industry_of)
    out: dict[str, pd.DataFrame] = {}
    for sid in frames:
        ind = industry_of.get(sid)
        peers = allm[(allm["ind"] == ind) & (allm["sid"] != sid)]
        if ind is None or peers["sid"].nunique() < 1:
            base = mtm[sid][["date"]].copy()
            base["imtm_5d_peer"] = float("nan")
            base["imtm_20d_peer"] = float("nan")
            out[sid] = base
            continue
        g = peers.groupby("date")[["m5", "m20"]].mean().reset_index()
        g.columns = ["date", "imtm_5d_peer", "imtm_20d_peer"]
        out[sid] = mtm[sid][["date"]].merge(g, on="date", how="left")
    return out
