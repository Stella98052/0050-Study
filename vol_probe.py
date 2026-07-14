# -*- coding: utf-8 -*-
"""realized_vol_20d 訊號性質裁決（v2.13，2026/7/15 預先宣告三檢定）。

用法：python vol_probe.py --holdings holdings.csv

背景：診斷A（vs 二元標籤，混池）Holm 後顯著（r=+0.071），
診斷B（vs 連續報酬，逐日橫斷面）不顯著——兩個待裁決假說：
    H_timing（審查）：訊號來自「跨時間市場狀態」非「個股相對強弱」
                     → 擇時訊號，兩層架構的依據
    H_artifact（本方）：label_up = 毛報酬 > 固定成本門檻，波動率放大
                     離散度即機械性提高跨門檻機率 → 對「均值」無資訊

【三檢定與判讀規則（先寫死，Holm m=3）】
P1 市場層級占比：realized_vol_20d 的變異中，由「當日全池均值」解釋的
   比例（R²）。>50% → 市場層級特徵（支持擇時方向）。僅描述，不進 Holm。
P2 擇時檢定：當日全池均值波動率(t) vs 0050 未來 N 日淨報酬，
   每隔 N 日獨立取樣，Spearman。顯著且方向穩定 → H_timing 成立可用。
P3 假象檢定（獨立子樣本）：
   corr(vol, fwd_return_net)＝均值效應 vs corr(vol, |fwd_return_net|)＝離散度效應
   離散度顯著而均值不顯著 → H_artifact 成立（診斷A的顯著性不可交易）。

結局矩陣（預先宣告）：
   P3均值✗離散✓ & P2✗ → 假象，歸檔，特徵搜索繼續（第四優先分產業）
   P3均值✗離散✓ & P2✓ → 二元關聯屬假象，但市場層級擇時訊號真實
                        → 兩層架構立項（擇時層），選股層維持無訊號結論
   P3均值✓            → 波動率對均值有資訊 → 直接進折內回歸+IC評估
只用開發集（斷言強制）；holdout 不碰。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from config.phase2_config import Phase2Config, PHASE2_VERSION
from src.fetch.holdings import load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.features.feature_matrix import (add_volatility_features,
                                         build_feature_matrix)
from src.model.walk_forward import holdout_start_date
from src.validate.vg3_significance import holm_correction
from feature_signal_audit import independent_subsample
# v2.14：三檢定收編為 VG-7 標準關卡，本腳本轉為薄殼（re-export 向後相容）
from src.validate.vg7_feature_probe import (market_level_share, timing_ic,   # noqa: F401
                                            mean_dispersion_effects,
                                            probe_feature_artifact)


def artifact_probe(sub, col="realized_vol_20d"):
    """向後相容別名 → vg7.mean_dispersion_effects。"""
    return mean_dispersion_effects(sub, col)


def main() -> int:
    print(f"▶ vol_probe（phase2 v{PHASE2_VERSION} / phase1 v{PHASE1_VERSION}）")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    args = ap.parse_args()
    if not args.holdings.exists():
        print(f"[中止] 找不到 {args.holdings.resolve()}")
        return 1

    p1, p2 = Config(), Phase2Config()
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    snap = load_holdings_from_csv(args.holdings, end)
    parts = []
    frames = {}
    for sid in snap.stock_ids:
        df = fetch_stock_history(sid, start, end, p1)
        frames[sid] = df
        f = build_feature_matrix(df, p1, p2)
        vol = add_volatility_features(df)
        for c in vol.columns:
            f[c] = vol[c].to_numpy()
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)
    feats["date"] = pd.to_datetime(feats["date"])
    hstart = pd.Timestamp(holdout_start_date(
        pd.DatetimeIndex(feats["date"].unique()), p2))
    dev = feats[feats["date"] < hstart].dropna(subset=["fwd_return_net"]).copy()
    assert dev["date"].max() < hstart, "probe 觸及 holdout"
    print(f"開發集 {len(dev)} 列（holdout 未觸及）")

    bench = fetch_stock_history("0050", start, end, p1)
    bench = bench[pd.to_datetime(bench["date"]) < hstart]

    p1_share = market_level_share(dev, "realized_vol_20d")
    print(f"\nP1 市場層級占比 R² = {p1_share:.1%}"
          f" → {'市場層級主導（支持擇時方向）' if p1_share > 0.5 else '個股層級成分仍高'}")

    t = timing_ic(dev, bench, p1, p2.forward_return_days, "realized_vol_20d")
    sub = independent_subsample(dev, p2.forward_return_days)
    a = artifact_probe(sub)

    _, adj = holm_correction([t["p"], a["p_mean"], a["p_disp"]])
    t_sig, mean_sig, disp_sig = (x < 0.05 for x in adj)
    print(f"P2 擇時：IC={t['ic']} p_raw={t['p']:.4f} p_holm={adj[0]:.4f} "
          f"n={t['n']} → {'✅顯著' if t_sig else '不顯著'}")
    print(f"P3 均值效應：r={a['r_mean']} p_holm={adj[1]:.4f} "
          f"→ {'✅顯著' if mean_sig else '不顯著'}")
    print(f"P3 離散度效應：r={a['r_disp']} p_holm={adj[2]:.4f} "
          f"→ {'✅顯著' if disp_sig else '不顯著'}（n={a['n']}）")

    print("\n===== 裁決（預先宣告結局矩陣）=====")
    if mean_sig:
        print("P3 均值效應顯著 → 波動率對報酬均值含資訊，"
              "進入 walk-forward 折內回歸+IC 評估。")
    elif disp_sig and t_sig:
        print("均值✗離散✓且擇時✓ → 診斷A的二元關聯屬離散度假象，"
              "但市場層級擇時訊號真實 → 兩層架構立項（擇時層）；"
              "選股層維持『無訊號』結論。")
    elif disp_sig:
        print("均值✗離散✓且擇時✗ → H_artifact 成立：診斷A之顯著為"
              "『二元標籤×固定成本門檻×離散度』機械假象，不可交易。"
              "realized_vol_20d 歸檔，特徵搜索續行第四優先（官方分類分產業）。")
    else:
        print("三檢定皆不顯著 → 無法定性，樣本內證據不足，歸檔待前瞻資料。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
