# -*- coding: utf-8 -*-
"""每日收盤後自動更新入口：PY 啟動即自動抓資料。

用法（Windows 建議掛「工作排程器」於每交易日 15:00 後執行）：
    python auto_update.py --holdings holdings.csv          # 更新日K（增量）
    python auto_update.py --holdings holdings.csv --quote  # 附帶即時/收盤快照

行為：
    1. 讀持股清單（CSV，10 檔檢查）
    2. 降級鏈抓日K：Fugle（若已設 FUGLE_API_KEY 環境變數）→ TWSE
       TWSE 路徑天然增量：已快取月份不重抓，只補當月
    3. 跑 VG-1 完整性檢查並印出全部警告
    4. 輸出 data/processed/{stock_id}_daily.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from src.fetch.auto_fetcher import build_default_chain, AllSourcesFailedError
from src.fetch.holdings import load_holdings_from_csv
from src.validate.trading_calendar import build_union_calendar
from src.validate.vg1 import print_report, run_vg1_validation, save_validation_report


def main() -> int:
    print(f"▶ 0050 phase1 v{PHASE1_VERSION}")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    ap.add_argument("--quote", action="store_true", help="附帶列印即時/收盤快照")
    ap.add_argument("--days", type=int, default=None,
                    help="回補天數（預設 Config.history_years 年）")
    args = ap.parse_args()

    if not args.holdings.exists():
        print(f"[中止] 找不到持股清單檔：{args.holdings.resolve()}")
        return 1

    cfg = Config()
    end = date.today()
    start = end - timedelta(days=args.days or cfg.history_years * 365)
    chain = build_default_chain(cfg)
    print(f"資料來源鏈：{' → '.join(s.name for s in chain._sources)}")

    snapshot = load_holdings_from_csv(args.holdings, end)
    frames = {}
    for sid in snapshot.stock_ids:
        try:
            frames[sid] = chain.daily(sid, start, end)
            src = frames[sid].attrs.get("data_source", "?")
            print(f"  {sid}: {len(frames[sid])} 筆（{src}）")
        except AllSourcesFailedError as exc:
            print(f"  ⚠ {sid}: 全部來源失敗 — {exc}")

    if not frames:
        print("[中止] 無任何股票抓取成功。")
        return 2

    calendar = build_union_calendar(frames)
    all_ok = True
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    for sid, df in frames.items():
        rpt = run_vg1_validation(df, sid, calendar, cfg)
        print_report(rpt)
        save_validation_report(rpt, Path("data"))
        all_ok &= rpt.passed
        df.to_csv(cfg.processed_dir / f"{sid}_daily.csv", index=False)

    if args.quote:
        for sid in frames:
            try:
                q = chain.quote(sid)
                shown = q.price if q.price is not None else f"—（昨收 {q.prev_close}）"
                print(f"  [快照] {sid}: {shown}  來源={q.data_source} {q.note}")
            except Exception as exc:                    # noqa: BLE001
                print(f"  [快照] {sid}: 失敗 — {exc!r}")

    if not all_ok:
        print("⚠ 部分股票 VG-1 未通過，請依警告人工確認。")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
