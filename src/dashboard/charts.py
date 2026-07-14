# -*- coding: utf-8 -*-
"""Plotly 圖表（規格：主圖 OHLC+波浪轉折雙色；副圖 MV 潮汐柱+潮汐線）。
純函式回傳 figure，無 streamlit 依賴（可單元測試）。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_candles_figure(tail: pd.DataFrame, pivots_retro, pivots_rt,
                        lookback_start) -> go.Figure:
    """K線 + 轉折標註：retrospective=灰（僅視覺參考，禁入模型），
    realtime=藍（可用於決策層）。"""
    fig = go.Figure(go.Candlestick(
        x=pd.to_datetime(tail["date"]), open=tail["open"], high=tail["high"],
        low=tail["low"], close=tail["close"], name="OHLC",
        increasing_line_color="#d33", decreasing_line_color="#2a2"))
    t0 = pd.Timestamp(lookback_start)
    for pv, color, tag in ((pivots_retro, "#999", "retro"),
                           (pivots_rt, "#1f77b4", "rt")):
        xs = [pd.Timestamp(p.pivot_date) for p in pv
              if pd.Timestamp(p.pivot_date) >= t0]
        ys = [p.price for p in pv if pd.Timestamp(p.pivot_date) >= t0]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+lines", name=f"pivots({tag})",
            line=dict(color=color, dash="dot", width=1),
            marker=dict(color=color, size=7)))
    fig.update_layout(height=430, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=28, b=10),
                      legend=dict(orientation="h"))
    return fig


def make_mv_figure(tail: pd.DataFrame, mv: pd.DataFrame,
                   lookback_start, cfg) -> go.Figure:
    """成交量柱 + 5/13/20MV 潮汐線；13MV 為核心否決線（定案2）粗線標示。"""
    m = mv.copy()
    m["date"] = pd.to_datetime(tail["date"].reset_index(drop=True)) \
        if len(m) == len(tail) else pd.to_datetime(m["date"])
    fig = go.Figure(go.Bar(x=pd.to_datetime(tail["date"]), y=tail["volume"],
                           name="volume", marker_color="#bbb"))
    joined = m.tail(len(tail))
    for col, name, width in (("mv_short", f"{cfg.vol_ma_short}MV", 1),
                             ("mv_mid", f"{cfg.vol_ma_mid}MV(核心否決線)", 3),
                             ("mv_long", f"{cfg.vol_ma_long}MV", 1)):
        if col in joined.columns:
            fig.add_trace(go.Scatter(x=pd.to_datetime(tail["date"]),
                                     y=joined[col].to_numpy(),
                                     name=name, line=dict(width=width)))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=24, b=10),
                      legend=dict(orientation="h"))
    return fig
