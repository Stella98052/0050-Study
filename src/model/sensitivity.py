# -*- coding: utf-8 -*-
"""ZigZag 閾值敏感度分析（3%–8%）。【明確標註：樣本內參數優化】
holdout 期間完全不參與本分析；最終驗證於獨立樣本外進行。◇熱圖僅視覺化。"""
from __future__ import annotations
import dataclasses
from pathlib import Path

import pandas as pd

from config.phase2_config import Phase2Config


def zigzag_sensitivity_analysis(ohlcv_by_stock: dict[str, pd.DataFrame],
                                p1_cfg, p2_cfg: Phase2Config,
                                benchmark_return_fn=None) -> pd.DataFrame:
    """逐閾值：重建特徵 → Walk-Forward（不含 holdout）→ 折平均 Sharpe/MDD。
    輸出 DataFrame(threshold, sharpe_net, mdd_net, n_trades, n_folds)。"""
    from src.features.feature_matrix import build_feature_matrix
    from src.model.walk_forward import run_walk_forward
    rows = []
    for th in p2_cfg.zigzag_grid:
        cfg_th = dataclasses.replace(p1_cfg, zigzag_threshold=th)
        feats = pd.concat(
            [build_feature_matrix(df, cfg_th, p2_cfg)
             for df in ohlcv_by_stock.values()], ignore_index=True
        ).sort_values("date").reset_index(drop=True)
        folds = run_walk_forward(feats, p2_cfg, benchmark_return_fn)
        if not folds:
            rows.append({"threshold": th, "sharpe_net": None, "mdd_net": None,
                         "n_trades": 0, "n_folds": 0})
            continue
        rows.append({
            "threshold": th,
            "sharpe_net": round(sum(f.metrics.sharpe_net for f in folds)
                                / len(folds), 4),
            "mdd_net": round(sum(f.metrics.mdd_net for f in folds)
                             / len(folds), 4),
            "n_trades": sum(f.metrics.n_trades for f in folds),
            "n_folds": len(folds),
        })
    out = pd.DataFrame(rows)
    print("【標註】以上為樣本內參數優化結果；最終評估以獨立 holdout 為準。")
    return out


def plot_sensitivity_heatmaps(result: pd.DataFrame, out_dir: Path) -> list[Path]:
    """◇ matplotlib 熱圖：閾值×Sharpe、閾值×MDD → PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for col, title in (("sharpe_net", "ZigZag threshold x Sharpe(net)"),
                       ("mdd_net", "ZigZag threshold x MDD(net)")):
        fig, ax = plt.subplots(figsize=(7, 2.2))
        vals = result[col].astype(float).to_numpy().reshape(1, -1)
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn"
                       if col == "sharpe_net" else "RdYlGn_r")
        ax.set_xticks(range(len(result)))
        ax.set_xticklabels([f"{t:.0%}" for t in result["threshold"]])
        ax.set_yticks([])
        ax.set_title(title + "  [in-sample]")
        for i, v in enumerate(result[col]):
            ax.text(i, 0, f"{v:.3f}" if pd.notna(v) else "–",
                    ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        p = out_dir / f"sensitivity_{col}.png"
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        paths.append(p)
    return paths
