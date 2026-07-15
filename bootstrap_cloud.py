# -*- coding: utf-8 -*-
"""雲端首次啟動引導（Streamlit Cloud 無本機快取時用）。

面板啟動時呼叫：若 data/models 無模型包或 data/raw 無快取，
以背景/有限逾時方式引導使用者先建置，而非讓面板卡死在抓取。
設計原則：寧可明確告知「尚未建置、請執行 X」，不假裝有資料。
"""
from __future__ import annotations

from pathlib import Path


def cloud_readiness(model_path: Path, holdings_path: Path) -> dict:
    """回傳雲端就緒狀態，供面板決定顯示哪種引導。"""
    has_model = model_path.exists()
    has_holdings = holdings_path.exists()
    # 快取粗略判定：data/raw 下任一股票目錄有 csv
    raw_dir = Path("data/raw")
    has_cache = raw_dir.exists() and any(raw_dir.rglob("*.csv"))
    return {
        "has_model": has_model,
        "has_holdings": has_holdings,
        "has_cache": has_cache,
        "ready": has_model and has_holdings,
    }


READINESS_GUIDE = """\
### 雲端尚未完成建置

此公開面板需要先產生模型包與資料快取才能完整運作。維護者請在本機執行：

```
python run_phase2.py --holdings holdings.csv    # 產生 data/models/model_phase2-v1.joblib
```

然後將 `data/models/` 與 `data/raw/`（資料快取）一併 commit 進 GitHub，
Streamlit Cloud 會自動重新部署。

**為何不在雲端即時抓取**：首次需抓十檔×十年 TWSE 官方資料（數分鐘），
雲端有記憶體與逾時限制，直接抓取會失敗——預先建置並 commit 快取是
最穩定的作法。技術圖（K線/波浪/MV）在有快取後即可離線呈現。
"""
