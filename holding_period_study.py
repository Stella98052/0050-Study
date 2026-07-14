# -*- coding: utf-8 -*-
"""持有期 N 網格研究（依 2026/7/12 審查建議：網格看趨勢形狀，非單點挑好看）。

用法：python holding_period_study.py --holdings holdings.csv
     （預設網格 5/10/15/20/30；資料走快取，不重新抓取）

針對「第3浪+潮汐爆發」訊號本身（不含模型層），逐 N 輸出（僅樣本內，
holdout 不參與）：
    交易層描述（全部交易均值）、獨立樣本數、隨機進場基準、邊際優勢、
    VG-3 p 值（v2.4 起僅餵統計獨立子集，定案4原文）、VG-4 蓋過判定。

v2.4 修正（外部審查 2026/7/13，經驗算採納）：
  ① VG-3 改餵 independent_return_series 篩出的獨立子集——先前餵入
     重疊交易層序列屬偽重複，四個 N 齊釘 p=0.0 即其人為假象（L13）
  ② VG-4 未達門檻時，VG-3 結果標「未決(VG-4蓋過)」，與 holdout 同一規則
  ③ 新增隨機進場基準 baseline = 同池同 N「任一日進場」平均淨報酬，
     邊際 edge = 訊號淨均 − baseline。多頭樣本中任何長多策略的淨報酬
     都會隨 N 上升（吃到市場飄移）；能支持假說的是 edge 隨 N 擴大，
     而非淨報酬由負轉正。

判讀準則（預先宣告，避免事後解釋；v2.5 依審查補三條）：
    ① edge 採「配對基準」：每筆獨立訊號的 edge_i = 訊號報酬 −
       同進場日全池 N 日均值（matched-date，逐筆扣除當時市場環境），
       並以 bootstrap 輸出 edge 均值之 95% CI——相鄰 N 的 CI 大幅重疊
       即為「雜訊連線」的客觀依據，不靠肉眼
    ② 趨勢結論資格：至少 3 個「相鄰」N 同時通過 VG-4 門檻，
       才允許對 edge 趨勢下結論；單一 N 過門檻只能評論該點本身
    ③ 多重比較（B1，v2.7）：逐 N 的 p 值須經 Holm 校正——m 次檢定中
       純雜訊至少一次假顯著機率 = 1−(0.95)^m（m=5 → 22.6%），
       單點 raw p 顯著不算數，以校正後 p 為準
    ④ 趨勢判定（B2，v2.7）：不得肉眼連線。edge 對 N 做線性迴歸，
       「斜率>0 且斜率 p<0.05」才允許宣稱「edge 隨 N 擴大」；
       zigzag/非單調形狀一律交由迴歸裁決
    ⑤ 假說成立需同時滿足：②資格 + ④斜率顯著為正 +（至少一點
       通過③校正後仍顯著）；缺一即為「未決」，不得選擇性引用好看的點
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from config.phase2_config import Phase2Config, PHASE2_VERSION
from src.fetch.holdings import load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.model.walk_forward import holdout_start_date
from src.validate.vg3_significance import permutation_test
from src.validate.vg4_sample import build_vg4_report

N_GRID = (5, 10, 15, 20, 30)


def trend_verdict(slope: float, pvalue: float) -> str:
    """趨勢判定措辭分級（v2.9）：臨界未顯著（0.05≤p<0.10）與明確否定
    必須在語氣上區隔——前者是擦邊球、後者是鐵證，份量不同（L19）。"""
    if slope > 0 and pvalue < 0.05:
        return "✅ 斜率顯著為正"
    if slope > 0 and pvalue < 0.10:
        return (f"❌ 斜率未達顯著（p={pvalue:.4f} 屬臨界未顯著，"
                "非明確否定）——不得宣稱趨勢，但證據強度低於明確否定")
    return "❌ 斜率未達顯著，不得宣稱 edge 隨 N 擴大"


def matched_edge_series(sig_ind: pd.Series, all_ret: pd.Series) -> pd.Series:
    """v2.5 配對 edge：edge_i = 獨立訊號報酬_i − 同進場日全池 N 日均值。

    all_ret 為全池各股逐日 N 日淨報酬（date index 可重複），
    先 groupby 日期取均值成「當日市場基準」，再與獨立訊號逐日對齊相減。
    對齊不到基準的日子（理論上不應發生）剔除該筆並保持可見。"""
    if len(sig_ind) == 0:
        return pd.Series(dtype="float64")
    mkt = all_ret.groupby(all_ret.index).mean()
    base = mkt.reindex(sig_ind.index)
    miss = int(base.isna().sum())
    if miss:
        print(f"  ⚠ 配對基準缺 {miss} 筆（訊號日不在全池報酬索引），該筆剔除")
    return (sig_ind - base).dropna()


def main() -> int:
    print(f"▶ holding_period_study（phase2 v{PHASE2_VERSION} / phase1 v{PHASE1_VERSION}）")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    args = ap.parse_args()
    if not args.holdings.exists():
        print(f"[中止] 找不到 {args.holdings.resolve()}")
        return 1

    p1 = Config()
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    snap = load_holdings_from_csv(args.holdings, end)
    frames = {}
    for sid in snap.stock_ids:
        try:
            frames[sid] = fetch_stock_history(sid, start, end, p1)
            print(f"  {sid}: {len(frames[sid])} 筆")
        except Exception as exc:                     # noqa: BLE001
            print(f"  ⚠ {sid}: 抓取失敗，本輪跳過（{exc!r}）")
    if len(frames) < 5:
        print("[中止] 成功抓取不足 5 檔，樣本無意義。")
        return 2
    print(f"股票池：成功 {len(frames)}/{len(snap.stock_ids)} 檔")
    all_dates = pd.DatetimeIndex(sorted(
        {d for df in frames.values() for d in pd.to_datetime(df["date"])}))

    from run_phase2 import _strategy_signal_returns
    rows = []
    for N in N_GRID:
        p2 = Phase2Config(forward_return_days=N)
        hstart = holdout_start_date(all_dates, p2)
        sig_ret, all_ret, sig_series = _strategy_signal_returns(
            frames, p1, p2, hstart, after=False)
        _, _, sig_oos = _strategy_signal_returns(
            frames, p1, p2, hstart, after=True)
        from src.signal_events import canonical_independent_samples
        from src.validate.vg3_significance import bootstrap_ci
        vg4 = build_vg4_report(sig_series, sig_oos, p2)
        _n, _kept, sig_ind = canonical_independent_samples(sig_series, sig_ret, N)
        assert _n == vg4.n_independent, "VG-3/VG-4 樣本來源不同步（不可能，同函式）"
        vg3 = permutation_test(sig_ind, all_ret, p2)   # v2.6：正典樣本，與VG-4同源
        # v2.5 配對基準：每筆獨立訊號 − 同進場日全池 N 日均值
        edge_series = matched_edge_series(sig_ind, all_ret)
        edge_boot = bootstrap_ci(edge_series, "mean_return", p2, N)
        edge_mean = float(edge_series.mean()) if len(edge_series) else 0.0
        if vg4.reliable:
            verdict = "顯著" if vg3.passed else "不顯著"
        else:
            verdict = "未決(VG-4蓋過)"                        # 與 holdout 同一規則
        rows.append({
            "N": N,
            "avg_net_indep%": round(float(sig_ind.mean()) * 100, 4)
                              if len(sig_ind) else None,
            "edge_mean%": round(edge_mean * 100, 4),
            "edge_ci_low%": (round(edge_boot.ci_low * 100, 4)
                             if edge_boot.ci_low is not None else None),
            "edge_ci_high%": (round(edge_boot.ci_high * 100, 4)
                              if edge_boot.ci_high is not None else None),
            "n_trades": int(len(sig_ret)),
            "n_indep": vg4.n_independent,
            "vg4_pass": vg4.reliable,
            "vg3_p_indep": vg3.p_value,
            "verdict": verdict,
        })
        rw = rows[-1]
        print(f"  N={N:>2}: 獨立均{rw['avg_net_indep%']:+.3f}% "
              f"配對edge{rw['edge_mean%']:+.3f}% "
              f"CI95[{rw['edge_ci_low%']}, {rw['edge_ci_high%']}] "
              f"獨立{vg4.n_independent} p={vg3.p_value} → {verdict}")

    # 規則③（B1）：Holm 多重比較校正
    from src.validate.vg3_significance import holm_correction
    raw_ps = [r_["vg3_p_indep"] for r_ in rows]
    rejects, adj_ps = holm_correction(raw_ps, alpha=0.05)
    for r_, rej, ap in zip(rows, rejects, adj_ps):
        r_["vg3_p_holm"] = ap
        if r_["verdict"] == "顯著" and not rej:
            r_["verdict"] = "校正後不顯著(Holm)"
        elif r_["verdict"] == "顯著" and rej:
            r_["verdict"] = "顯著(Holm後仍成立)"
    print("Holm 校正後 p：" + "  ".join(
        f"N={r_['N']}:{r_['vg3_p_holm']}" for r_ in rows))

    # 規則④（B2）：edge 對 N 線性迴歸趨勢檢定（禁止肉眼連線）
    from scipy import stats as _st
    import numpy as _np
    Ns = _np.array([r_["N"] for r_ in rows], dtype=float)
    edges = _np.array([r_["edge_mean%"] for r_ in rows], dtype=float)
    reg = _st.linregress(Ns, edges)
    trend_ok = (reg.slope > 0) and (reg.pvalue < 0.05)
    print(f"趨勢迴歸：斜率={reg.slope:.4f}%/日  p={reg.pvalue:.4f}  "
          f"R²={reg.rvalue**2:.4f} → {trend_verdict(reg.slope, reg.pvalue)}")

    # 規則②：至少 3 個相鄰 N 同過 VG-4 才有趨勢結論資格
    passes = [r["vg4_pass"] for r in rows]
    run_len = cur = 0
    for p_ in passes:
        cur = cur + 1 if p_ else 0
        run_len = max(run_len, cur)
    if run_len >= 3:
        print("趨勢結論資格：✅ 有（≥3 相鄰 N 通過 VG-4），可對 edge 趨勢下結論")
    else:
        print(f"趨勢結論資格：❌ 無（最長相鄰通過串={run_len} <3）——"
              f"edge 趨勢僅為線索，不得寫入結論；優先擴大樣本分母")

    out = pd.DataFrame(rows)
    p2 = Phase2Config()
    p2.report_dir.mkdir(parents=True, exist_ok=True)
    path = p2.report_dir / "holding_period_study.csv"
    out.to_csv(path, index=False)
    print("【標註】本研究為樣本內探索（holdout 未參與）；"
          "VG-4 門檻隨 N 變嚴屬設計行為，判讀需兩線並看。")
    print(f"結果已存 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
