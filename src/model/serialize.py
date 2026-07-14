# -*- coding: utf-8 -*-
"""模型序列化（joblib）：模型 + 特徵清單 + Config + 版本標記，供第三階段載入。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib

from config.phase2_config import Phase2Config, PHASE2_VERSION


@dataclass(frozen=True)
class ModelBundle:
    feature_names: tuple[str, ...]
    zigzag_threshold_used: float
    trained_at: str
    version_tag: str
    vg_summary: dict            # VG-1~VG-5 通過狀態（第三階段面板小卡）
    p2_config: dict             # asdict(Phase2Config)（Path 轉 str）


def save_bundle(model, bundle: ModelBundle, cfg: Phase2Config) -> Path:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.model_dir / f"model_{bundle.version_tag}.joblib"
    joblib.dump({"model": model, "bundle": asdict(bundle),
                 "phase2_version": PHASE2_VERSION}, path)
    return path


def load_bundle(path: Path):
    """載入並校驗：特徵欄清單存在且非空；欄位不符由呼叫端比對後明確 raise。"""
    payload = joblib.load(path)
    bundle = ModelBundle(**{k: (tuple(v) if k == "feature_names" else v)
                            for k, v in payload["bundle"].items()})
    if not bundle.feature_names:
        raise ValueError("模型包缺少特徵清單，拒絕載入（不靜默套用）")
    return payload["model"], bundle


def assert_feature_alignment(bundle: ModelBundle, current_cols: list[str]) -> None:
    missing = [c for c in bundle.feature_names if c not in current_cols]
    if missing:
        raise ValueError(f"特徵欄位不一致，缺少 {missing}；拒絕套用模型（不靜默）")
