# -*- coding: utf-8 -*-
"""診斷工具：找出重複日期的來源快取檔（契合「矛盾優先診斷」原則）。

用法：
    python diagnose_duplicates.py 2454
    python diagnose_duplicates.py 2454 2308

逐檔掃描 data/raw/{stock_id}/*.csv，回報：
    1. 檔內越月列：檔名月份與資料列月份不符（最可能的重複根因）
    2. 跨檔重複：同一日期出現在多個快取檔
    3. 檔內重複：同一檔內同日期多列
輸出每筆異常的「檔名 → 日期」對應，供人工核對後刪除該月快取重抓。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.config import Config, PHASE1_VERSION  # noqa: E402


def diagnose(stock_id: str, cfg: Config) -> None:
    folder = cfg.cache_dir / stock_id
    if not folder.exists():
        print(f"[{stock_id}] 找不到快取資料夾：{folder.resolve()}")
        return
    print(f"\n===== 診斷 {stock_id}（{folder.resolve()}）=====")
    date_to_files: dict[str, list[str]] = defaultdict(list)
    issues = 0

    for csv_path in sorted(folder.glob("*.csv")):
        yyyymm = csv_path.stem
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
        except Exception as exc:                     # noqa: BLE001
            print(f"  ✗ {csv_path.name} 無法讀取：{exc!r}")
            issues += 1
            continue
        if len(df) == 0:
            continue

        # 1) 越月列
        y, m = int(yyyymm[:4]), int(yyyymm[4:])
        out = df[(df["date"].dt.year != y) | (df["date"].dt.month != m)]
        if len(out):
            months = sorted(out["date"].dt.strftime("%Y%m").unique())
            print(f"  ✗ {csv_path.name} 含 {len(out)} 筆越月列"
                  f"（實際月份：{', '.join(months)}）← 最可能的重複根因，"
                  f"建議刪除此檔後重跑（會自動重抓該月）")
            issues += 1

        # 3) 檔內重複
        n_in = int(df["date"].duplicated().sum())
        if n_in:
            print(f"  ✗ {csv_path.name} 檔內重複日期 {n_in} 筆")
            issues += 1

        for d in df["date"].dt.date:
            date_to_files[str(d)].append(csv_path.name)

    # 2) 跨檔重複
    cross = {d: fs for d, fs in date_to_files.items() if len(fs) > 1}
    if cross:
        print(f"  ✗ 跨檔重複日期 {len(cross)} 個：")
        for d, fs in sorted(cross.items())[:15]:
            print(f"      {d} ← {' + '.join(sorted(set(fs)))}")
        if len(cross) > 15:
            print(f"      …其餘 {len(cross) - 15} 個省略")
        issues += 1

    if issues == 0:
        print("  ✓ 未發現越月列、檔內重複或跨檔重複。")
    else:
        print(f"  共 {issues} 類異常。處置：刪除被點名的 .csv 檔 → 重跑 "
              f"run_phase1.py（僅重抓被刪月份，其餘快取不動）。")


if __name__ == "__main__":
    print(f"▶ 0050 phase1 v{PHASE1_VERSION}")
    targets = sys.argv[1:] or ["2454", "2308"]
    cfg = Config()
    for sid in targets:
        diagnose(sid, cfg)
