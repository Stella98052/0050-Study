# -*- coding: utf-8 -*-
"""Walk-Forward 驗證：訓練3年/測試3月/步長3月/Embargo 30日；
最後 holdout_months 完全保留（不進任何折，供最終樣本外驗證）。⛔每折斷言。"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from config.phase2_config import Phase2Config


@dataclass(frozen=True)
class WalkForwardSplit:
    fold_id: int
    train_start: date
    train_end: date
    embargo_end: date
    test_start: date
    test_end: date


def holdout_start_date(all_dates: pd.DatetimeIndex, cfg: Phase2Config) -> date:
    """最終樣本外起點 = 最末日 − holdout_months。"""
    return (all_dates.max() - pd.DateOffset(months=cfg.holdout_months)).date()


def generate_walk_forward_splits(
    all_dates: pd.DatetimeIndex, cfg: Phase2Config
) -> list[WalkForwardSplit]:
    d0 = all_dates.min().date()
    hstart = holdout_start_date(all_dates, cfg)
    splits: list[WalkForwardSplit] = []
    fold = 0
    cursor = d0
    while True:
        tr_start = cursor
        tr_end = (pd.Timestamp(tr_start) + pd.DateOffset(years=cfg.train_window_years)
                  ).date()
        emb_end = tr_end + timedelta(days=cfg.embargo_days)
        te_start = emb_end + timedelta(days=1)
        te_end = (pd.Timestamp(te_start) + pd.DateOffset(months=cfg.test_window_months)
                  ).date()
        if te_end >= hstart:                     # 測試期不得侵入 holdout
            break
        splits.append(WalkForwardSplit(fold, tr_start, tr_end, emb_end,
                                       te_start, te_end))
        fold += 1
        cursor = (pd.Timestamp(cursor) + pd.DateOffset(months=cfg.step_months)).date()
    return splits


def run_walk_forward(features: pd.DataFrame, p2_cfg: Phase2Config,
                     benchmark_return_fn=None) -> list:
    """★⛔ 逐折：訓練→測試期預測→交易報酬（預測為多→以該列標籤報酬成交）。

    benchmark_return_fn(start,end)->float：同期 0050 買進持有報酬（算 Alpha）；
    None 時 Alpha 以 0 基準（測試用）。回傳 FoldResult 清單。
    """
    from src.features.feature_matrix import report_class_balance
    from src.model.metrics import compute_backtest_metrics, FoldResult
    from src.model.train import get_feature_importance, train_model
    from src.validate.vg5_asserts import vg5_assert_train_test_no_overlap

    dts = pd.to_datetime(features["date"]).dt.date
    valid = features["fwd_return_gross"].notna()
    results = []
    for sp in generate_walk_forward_splits(
            pd.DatetimeIndex(pd.to_datetime(features["date"].unique())), p2_cfg):
        vg5_assert_train_test_no_overlap(sp)
        tr = features[(dts >= sp.train_start) & (dts <= sp.train_end) & valid]
        te = features[(dts >= sp.test_start) & (dts <= sp.test_end) & valid]
        if len(tr) < 100 or len(te) < 10:
            continue
        bal = report_class_balance(tr["label_up"])
        model, feat_names = train_model(tr, p2_cfg, bal["scale_pos_weight"])
        proba = model.predict_proba(te)[:, 1]      # 包裝模型內部統一編碼
        picked = te[proba > 0.5]
        bench = (benchmark_return_fn(sp.test_start, sp.test_end)
                 if benchmark_return_fn else 0.0)
        metrics = compute_backtest_metrics(
            picked["fwd_return_gross"], picked["fwd_return_net"],
            dates=picked["date"],
            benchmark_return=bench, holding_days=p2_cfg.forward_return_days)
        results.append(FoldResult(
            split=sp, metrics=metrics, class_balance=bal,
            feature_importance=get_feature_importance(model, feat_names)))
    return results

