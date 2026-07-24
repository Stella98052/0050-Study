# -*- coding: utf-8 -*-
"""財務因子（v3.27）：EPS 加速度、三率改善、ROE、自由現金流。

輸入為 mops_financials.fetch_financials 的季表（新到舊，已過公告期限）。
所有比率以「同季 YoY」比較，避開台股季節性；缺季一律 None 不猜。
"""
from __future__ import annotations

import pandas as pd


_FLOW_COLS = ("revenue", "gross_profit", "op_income", "net_income",
              "ocf", "capex", "eps")          # 流量欄（累計）；equity 為存量


def to_single_quarter(fin: pd.DataFrame) -> pd.DataFrame:
    """MOPS 季報為**年初至該季累計數**，轉為單季數（L64）。

    實證（2330 實抓 2026/7/24）：2025 Q2/Q3/Q4 營收 1.77/2.76/3.81 兆，
    逐季遞增即累計；若直接相加四季＝重複計算（ROE 會膨脹逾一倍）。
    轉換：單季 = 本季累計 − 同年上季累計（Q1 累計即單季）。
    equity（權益）為資產負債表存量，不轉換。
    """
    if fin is None or len(fin) == 0:
        return fin
    d = fin.sort_values(["year", "season"]).reset_index(drop=True).copy()
    for col in _FLOW_COLS:
        if col not in d.columns:
            continue
        vals = []
        for i, r in d.iterrows():
            if int(r["season"]) == 1 or pd.isna(r[col]):
                vals.append(r[col])
                continue
            prev = d[(d["year"] == r["year"]) &
                     (d["season"] == int(r["season"]) - 1)]
            if len(prev) == 0 or pd.isna(prev.iloc[0][col]):
                vals.append(float("nan"))      # 缺上季 → 不猜
            else:
                vals.append(float(r[col]) - float(prev.iloc[0][col]))
        d[col] = vals
    return d.sort_values(["year", "season"], ascending=False).reset_index(
        drop=True)


def _same_season_prev_year(df: pd.DataFrame, y: int, s: int) -> pd.Series | None:
    m = df[(df["year"] == y - 1) & (df["season"] == s)]
    return m.iloc[0] if len(m) else None


def eps_growth_and_accel(fin: pd.DataFrame) -> tuple[float | None, float | None]:
    """(最新季 EPS YoY, EPS 加速度＝本季 YoY − 上季 YoY)。

    加速度為正＝成長正在加快（起漲的核心訊號之一）。
    """
    if fin is None or len(fin) == 0 or "eps" not in fin.columns:
        return None, None
    d = to_single_quarter(fin)             # 累計→單季（L64）
    yoys = []
    for i in range(min(2, len(d))):
        cur = d.iloc[i]
        base = _same_season_prev_year(d, int(cur["year"]), int(cur["season"]))
        if base is None or pd.isna(cur.get("eps")) or pd.isna(base.get("eps")):
            yoys.append(None)
            continue
        b = float(base["eps"])
        yoys.append((float(cur["eps"]) - b) / abs(b) if b != 0 else None)
    g = yoys[0] if yoys else None
    accel = (yoys[0] - yoys[1]) if len(yoys) >= 2 and yoys[0] is not None \
        and yoys[1] is not None else None
    return g, accel


def three_margins(fin: pd.DataFrame) -> dict:
    """三率（毛利率／營益率／淨利率）與其 YoY 變化（百分點）。"""
    out = {"gross": None, "op": None, "net": None,
           "gross_yoy": None, "op_yoy": None, "net_yoy": None}
    if fin is None or len(fin) == 0 or "revenue" not in fin.columns:
        return out
    d = to_single_quarter(fin)             # 累計→單季（L64）
    cur = d.iloc[0]
    rev = cur.get("revenue")
    if pd.isna(rev) or float(rev) == 0:
        return out
    pairs = (("gross", "gross_profit"), ("op", "op_income"),
             ("net", "net_income"))
    for key, col in pairs:
        if col in d.columns and pd.notna(cur.get(col)):
            out[key] = float(cur[col]) / float(rev)
    base = _same_season_prev_year(d, int(cur["year"]), int(cur["season"]))
    if base is not None and pd.notna(base.get("revenue")) and \
            float(base["revenue"]) != 0:
        for key, col in pairs:
            if col in d.columns and out[key] is not None and \
                    pd.notna(base.get(col)):
                out[f"{key}_yoy"] = out[key] - float(base[col]) / float(
                    base["revenue"])
    return out


def roe_ttm(fin: pd.DataFrame) -> float | None:
    """ROE＝近四季稅後淨利合計 ÷ 最新權益總額（不足四季則年化）。"""
    if fin is None or len(fin) == 0 or "net_income" not in fin.columns \
            or "equity" not in fin.columns:
        return None
    d = to_single_quarter(fin)             # 累計→單季（L64）
    ni = d["net_income"].dropna()
    eq = d["equity"].dropna()
    if len(ni) == 0 or len(eq) == 0 or float(eq.iloc[0]) == 0:
        return None
    n = min(4, len(ni))
    total = float(ni.head(n).sum()) * (4.0 / n)            # 年化
    return total / float(eq.iloc[0])


def free_cash_flow(fin: pd.DataFrame) -> float | None:
    """自由現金流＝營業活動現金流 − 資本支出（近四季合計）。"""
    if fin is None or len(fin) == 0 or "ocf" not in fin.columns:
        return None
    d = to_single_quarter(fin)             # 累計→單季（L64）
    ocf = d["ocf"].dropna()
    if len(ocf) == 0:
        return None
    capex = d["capex"].dropna() if "capex" in d.columns else pd.Series(
        dtype="float64")
    n = min(4, len(ocf))
    fcf = float(ocf.head(n).sum()) - float(abs(capex.head(n).sum())
                                           if len(capex) else 0.0)
    return fcf
