# -*- coding: utf-8 -*-
"""CI 環境自檢（審查採納，2026/7/15）：核心相依缺漏時「最先失敗、訊息直接」，
比從 LightGBMError 反推快。

設計強化（避免重蹈「兩處清單各自維護」覆轍）：
- import 對應表從 requirements.txt 動態解析套件名，測試不另存硬編碼清單
- 用 importlib.util.find_spec（只查存在、不執行套件初始化，快且無副作用）
- pkg→import 名對應（如 scikit-learn→sklearn）集中一處維護
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# 套件發行名 → import 名（僅列名稱不一致者；其餘同名自動處理）
_DIST_TO_IMPORT = {
    "scikit-learn": "sklearn",
    "beautifulsoup4": "bs4",
    "beautifulsoup4": "bs4",
    "beautifulsoup4": "bs4",
    "pytest": "pytest",
}
# 純測試/繪圖類，執行期核心邏輯非必需 → 自檢不強制（缺了不擋核心）
_OPTIONAL = {"matplotlib", "plotly", "streamlit", "pytest"}

_REQ = Path(__file__).resolve().parent.parent / "requirements.txt"


def _parse_requirements() -> list[str]:
    """解析 requirements.txt 的發行名（去版本/註解/空行）。"""
    names = []
    for line in _REQ.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 去掉版本規格與行末註解
        dist = line.split("#")[0].strip()
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            dist = dist.split(sep)[0]
        dist = dist.strip()
        if dist:
            names.append(dist)
    return names


def test_requirements_file_exists():
    """requirements.txt 必須存在（部署/CI 的單一相依來源）。"""
    assert _REQ.exists(), f"找不到 {_REQ}"


def test_core_packages_importable():
    """核心執行期相依（排除純測試/繪圖類）必須可匯入；
    缺漏時本測試最先失敗，直接點名缺哪個套件，不必反推 LightGBMError。"""
    missing = []
    for dist in _parse_requirements():
        if dist in _OPTIONAL:
            continue
        import_name = _DIST_TO_IMPORT.get(dist, dist.replace("-", "_"))
        if importlib.util.find_spec(import_name) is None:
            missing.append(f"{dist}（import 名：{import_name}）")
    assert not missing, (
        "CI/環境缺少必要套件：" + "、".join(missing) +
        "。請確認 requirements.txt 與安裝步驟一致（本測試自 requirements 解析，"
        "不另維護硬編碼清單）。")


def test_lightgbm_sklearn_interface_available():
    """專項守衛本次事故：LGBMClassifier 需 scikit-learn（lightgbm.sklearn）。
    直接驗證該介面可用，讓此類缺漏在專屬測試明確暴露。"""
    assert importlib.util.find_spec("sklearn") is not None, \
        "缺 scikit-learn：LightGBM 的 LGBMClassifier（sklearn 介面）將無法使用"
    import lightgbm as lgb
    assert hasattr(lgb, "LGBMClassifier"), "lightgbm.sklearn 介面不可用"
