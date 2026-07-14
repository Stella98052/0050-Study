# -*- coding: utf-8 -*-
"""模型服務（規格第三階段③）：載入 phase2 序列化模型包、校驗特徵欄一致
後才套用；欄位不符明確 raise，不靜默套用。"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.features.feature_matrix import FEATURE_COLS
from src.model.serialize import load_bundle


def load_model_pack(model_path: Path):
    """回傳 (model, bundle)；缺檔回 (None, None)（面板顯示占位，不假裝有模型）。"""
    if not model_path.exists():
        return None, None
    return load_bundle(model_path)


def predict_latest(model, bundle, features_df: pd.DataFrame) -> dict | None:
    """對特徵矩陣最後一列輸出未來 N 日看漲機率。

    先 assert_feature_alignment（欄位不符 → raise，由呼叫端呈現錯誤），
    僅取最後一列避免整表推論成本。"""
    if model is None or bundle is None or len(features_df) == 0:
        return None
    # 對齊原始 FEATURE_COLS（wrapper.predict_proba 內部會 one-hot 波浪標籤，
    # 故此處驗證原始欄齊備，而非展開後的 wave_1…；否則每檔都誤報缺欄）
    missing = [c for c in FEATURE_COLS if c not in features_df.columns]
    if missing:
        raise ValueError(f"特徵矩陣缺少原始欄位 {missing}；拒絕套用模型（不靜默）")
    last = features_df.tail(1)
    proba = float(model.predict_proba(last)[:, 1][0])
    return {"proba_up": round(proba, 4), "pick": proba > 0.5,
            "as_of": str(pd.to_datetime(last["date"].iloc[0]).date()),
            "model_tag": bundle.version_tag}
