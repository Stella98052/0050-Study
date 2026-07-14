# -*- coding: utf-8 -*-
"""VG-5 訊號來源正確性：管線關鍵節點的自動化斷言（程式實際執行，非文件聲明）。"""
from __future__ import annotations
import numpy as np
import pandas as pd

LABEL_COLS = ("entry_open_next", "fwd_return_gross", "fwd_return_net", "label_up")
FORBIDDEN_SUBSTRINGS = ("retrospective", "retro_")


def vg5_assert_no_retrospective(df: pd.DataFrame) -> None:
    """⛔ 特徵欄不得含任何 retrospective 來源欄（欄名黑名單）。違反→AssertionError。"""
    bad = [c for c in df.columns
           if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]
    assert not bad, f"VG-5 違反：偵測到 retrospective 來源欄位 {bad}，禁入訓練管線"


def vg5_assert_feature_before_label(
    ohlcv: pd.DataFrame, features: pd.DataFrame,
    p1_cfg, p2_cfg, n_samples: int = 3,
) -> None:
    """⛔ 防未來函數之「截斷重算」斷言（程式化執行）：

    隨機抽 n_samples 個時點 t，僅用 ohlcv[:t+1] 重算特徵，
    斷言與全量計算之第 t 列完全一致（未來資料不影響特徵）。
    另斷言：末 N 列標籤必為 NaN（標籤依賴未來窗，結構性證據）。
    """
    from src.features.feature_matrix import _compute_features_only
    n = len(features)
    N = p2_cfg.forward_return_days
    tail = features["fwd_return_gross"].iloc[-N:]
    assert tail.isna().all(), "VG-5 違反：末 N 列標籤非 NaN（標籤未依賴未來窗？）"

    rng = np.random.default_rng(p2_cfg.random_state)
    check_cols = ["mv_short", "mv_bias", "rsi_14", "macd_hist",
                  "ret_5d", "wave_label_realtime"]
    for t in rng.choice(np.arange(max(60, n // 3), n - N), size=n_samples,
                        replace=False):
        truncated = _compute_features_only(
            ohlcv.iloc[: t + 1].reset_index(drop=True), p1_cfg, p2_cfg)
        full_row = features.iloc[t]
        trunc_row = truncated.iloc[-1]
        for c in check_cols:
            a, b = full_row[c], trunc_row[c]
            same = (a == b) or (pd.isna(a) and pd.isna(b)) or (
                isinstance(a, float) and isinstance(b, float)
                and abs(a - b) < 1e-9)
            assert same, (f"VG-5 違反：t={t} 欄 {c} 截斷重算不一致 "
                          f"(full={a}, truncated={b}) → 特徵含未來資訊")


def vg5_assert_train_test_no_overlap(split) -> None:
    """⛔ train_end + embargo < test_start（邊界洩漏防護）。"""
    assert split.embargo_end < split.test_start, (
        f"VG-5 違反：fold {split.fold_id} embargo_end({split.embargo_end}) "
        f"未早於 test_start({split.test_start})")
    assert split.train_end <= split.embargo_end
