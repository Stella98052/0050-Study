# -*- coding: utf-8 -*-
"""技術指標（明確窗口；rolling/ewm 僅用截至當日資料，天然無未來函數）。★"""
from __future__ import annotations
import pandas as pd


def rsi(close: pd.Series, window: int) -> pd.Series:
    """★ Wilder RSI（窗口預設14）。
    calc_logic：漲跌分離 → Wilder 平滑（ewm alpha=1/window）→ 100−100/(1+RS)。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0).astype("float64").rename("rsi")


def macd(close: pd.Series, fast: int, slow: int, signal: int
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """★ MACD = EMA(fast) − EMA(slow)；訊號線 = EMA(MACD, signal)；柱 = 差。"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    m = (ema_f - ema_s).rename("macd")
    sig = m.ewm(span=signal, adjust=False).mean().rename("macd_signal")
    hist = (m - sig).rename("macd_hist")
    return m, sig, hist
