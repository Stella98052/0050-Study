# -*- coding: utf-8 -*-
"""前瞻預測紀錄（Model v2 最終閘的資料載體；規格第三階段①的延伸）。

每日收盤後 append 一列/檔；同 (stock_id, last_bar_date) 去重（重跑不重複計）。
裁決規則（預先宣告）：獨立樣本（間隔≥N，select_independent_dates）累積
≥ min_prospective_samples 後才做一次性統計裁決，期間不得偷看下結論。"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

COLS = ["run_ts", "stock_id", "last_bar_date", "close",
        "proba_up", "pick", "forward_days", "model_tag"]


def append_prediction(row: dict, path: Path) -> bool:
    """寫入一筆；重複鍵回 False（不覆蓋，保留首次紀錄的不可竄改性）。"""
    new = pd.DataFrame([{c: row[c] for c in COLS}])
    if path.exists():
        old = pd.read_csv(path, dtype={"stock_id": str})
        dup = ((old["stock_id"] == str(row["stock_id"])) &
               (old["last_bar_date"] == str(row["last_bar_date"]))).any()
        if dup:
            return False
        out = pd.concat([old, new], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = new
    out.to_csv(path, index=False)
    return True


def prospective_progress(path: Path, n_days: int) -> dict:
    """目前累積之『獨立』樣本數（逐檔間隔≥N 後彙總）與紀錄總數。"""
    from src.signal_events import select_independent_dates
    if not path.exists():
        return {"n_rows": 0, "n_independent": 0}
    df = pd.read_csv(path, dtype={"stock_id": str})
    n_ind = 0
    for _sid, g in df.groupby("stock_id"):
        dts = sorted(pd.to_datetime(g["last_bar_date"]))
        n_ind += len(select_independent_dates(dts, n_days))
    return {"n_rows": int(len(df)), "n_independent": int(n_ind)}


def load_predictions_view(path, stock_id: str | None = None):
    """讀取前瞻紀錄供面板呈現：全部或單股。回傳 DataFrame（空檔回空表）。

    此資料由 GitHub Actions「每日前瞻更新」每交易日自動累積並 commit，
    面板部署後每次重新載入即讀到最新（Streamlit Cloud 隨 repo push 更新）。"""
    import pandas as pd
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["last_bar_date"] = pd.to_datetime(df["last_bar_date"])
    df = df.sort_values("last_bar_date")
    if stock_id:
        df = df[df["stock_id"] == str(stock_id)]
    return df.reset_index(drop=True)


def latest_prediction_per_stock(path):
    """各股最近一筆預測（面板總覽表用）。"""
    import pandas as pd
    df = load_predictions_view(path)
    if len(df) == 0:
        return pd.DataFrame(columns=COLS)
    return (df.sort_values("last_bar_date")
              .groupby("stock_id", as_index=False).tail(1)
              .sort_values("stock_id").reset_index(drop=True))
