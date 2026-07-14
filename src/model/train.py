# -*- coding: utf-8 -*-
"""模型訓練（定案：LightGBM；固定 random_state 可重現）。★"""
from __future__ import annotations
import pandas as pd

from config.phase2_config import Phase2Config
from src.features.feature_matrix import FEATURE_COLS

_WAVE_CATS = ["1", "2", "3", "4", "5", "A", "B", "C", "unknown"]


def _encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """波浪標籤 one-hot（固定類別集，確保訓練/預測欄位一致）；布林轉 int。"""
    X = df[FEATURE_COLS].copy()
    wave = pd.Categorical(X.pop("wave_label_realtime"), categories=_WAVE_CATS)
    dummies = pd.get_dummies(wave, prefix="wave").astype("int64")
    dummies.index = X.index
    X = pd.concat([X, dummies], axis=1)
    for c in ("is_volume_burst", "mv_mid_veto_active"):
        X[c] = X[c].astype("int64")
    return X, list(X.columns)


def train_model(train_df: pd.DataFrame, cfg: Phase2Config,
                scale_pos_weight: float = 1.0):
    """★ 訓練 LGBMClassifier；回傳 (包裝後模型, 特徵名清單)。
    包裝物件之 predict_proba 接受原始特徵 df（內部套同一編碼）。"""
    import lightgbm as lgb
    X, names = _encode(train_df)
    y = train_df["label_up"].astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth, random_state=cfg.random_state,
        scale_pos_weight=scale_pos_weight if scale_pos_weight not in (0, float("inf")) else 1.0,
        verbose=-1,
    )
    clf.fit(X, y)
    return _Wrapped(clf, names), names


class _Wrapped:
    """包裝：predict_proba 接受含 FEATURE_COLS 的原始 df，內部統一編碼。"""

    def __init__(self, clf, names):
        self._clf, self._names = clf, names

    def predict_proba(self, df_or_x):
        X = df_or_x
        if "wave_label_realtime" in getattr(df_or_x, "columns", []):
            X, _ = _encode(df_or_x)
        return self._clf.predict_proba(X[self._names])

    @property
    def booster(self):
        return self._clf


def get_feature_importance(model, feat_names: list[str]) -> dict[str, float]:
    """★ 特徵重要性（gain），避免黑盒；輸出於報告與第三階段面板。"""
    imp = model.booster.feature_importances_
    total = imp.sum() or 1
    return {n: round(float(v) / total, 4)
            for n, v in sorted(zip(feat_names, imp), key=lambda t: -t[1])}
