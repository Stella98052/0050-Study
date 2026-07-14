# -*- coding: utf-8 -*-
"""第一階段執行入口：持股清單 → 逐月抓取 → VG-1 → 波浪 → 量能 → 訊號。

用法：
    python run_phase1.py                      # 白名單來源抓持股（可能需 CSV fallback）
    python run_phase1.py --holdings my.csv    # 以人工 CSV 覆寫持股清單

輸出：
    data/processed/{stock_id}_features.csv     日線 + realtime 波浪標籤 + MV 特徵 + 訊號
    data/validation_report_{stock_id}.json     VG-1 報告（含日曆輔助校驗警告）
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from src.fetch.holdings import fetch_0050_top10, load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.schemas import HoldingsUnavailableError
from src.signal_events import (
    build_signal_count_report, detect_wave3_tidal_burst,
)
from src.validate.trading_calendar import (
    build_union_calendar, cross_check_calendar, fetch_twse_holidays,
)
from src.validate.vg1 import print_report, run_vg1_validation, save_validation_report
from src.volume.volume_features import (
    compute_mv_features, detect_price_volume_divergence,
)
from src.wave.wave_labels import label_waves_realtime


def main() -> int:
    print(f"▶ 0050 phase1 v{PHASE1_VERSION}")
    print(DISCLAIMER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdings", type=Path, default=None,
                        help="人工覆寫持股清單 CSV（欄位：stock_id）")
    args = parser.parse_args()

    cfg = Config()
    end = date.today()
    start = end - timedelta(days=cfg.history_years * 365)

    # ---- 持股清單（白名單來源 → 失敗則要求 CSV fallback，禁止捏造）----
    if args.holdings:
        if not args.holdings.exists():
            print(f"[中止] 找不到持股清單檔：{args.holdings.resolve()}")
            print("       請確認檔案路徑，或建立 CSV（欄位：stock_id，共 10 列），")
            print("       並註明清單基準日與出處（例：元大投信官網持股比重頁）。")
            return 1
        snapshot = load_holdings_from_csv(args.holdings, end)
    else:
        try:
            snapshot = fetch_0050_top10(cfg)
        except HoldingsUnavailableError as exc:
            print(f"[中止] {exc}")
            return 1
    print(f"持股清單來源={snapshot.source} 基準日={snapshot.snapshot_date} "
          f"manual_override={snapshot.is_manual_override}")

    # ---- 逐股抓取（逐月 + 快取 + 重試）----
    frames: dict[str, pd.DataFrame] = {}
    for sid in snapshot.stock_ids:
        print(f"抓取 {sid} …")
        frames[sid] = fetch_stock_history(sid, start, end, cfg)

    # ---- 交易日曆（定案 1：聯集 + 官方休市日輔助校驗）----
    calendar = build_union_calendar(frames)
    holidays = fetch_twse_holidays(cfg)          # None → 降級並輸出警告
    cal_warnings = tuple(cross_check_calendar(calendar, holidays, start, end))

    # ---- 逐股：VG-1 → realtime 波浪 → MV → 背離 → 訊號 → 雙軌計數 ----
    all_passed = True
    for sid, df in frames.items():
        report = run_vg1_validation(df, sid, calendar, cfg, extra_warnings=cal_warnings)
        print_report(report)
        save_validation_report(report, Path("data"))
        all_passed &= report.passed

        wave_rt = label_waves_realtime(df, cfg)
        mv = compute_mv_features(df, cfg)
        div = detect_price_volume_divergence(df, mv, wave_rt)
        signal = detect_wave3_tidal_burst(wave_rt, mv, div, cfg)
        count = build_signal_count_report(signal, cfg)
        print(f"  [{sid}] {count.note}")

        out = df.reset_index(drop=True).join(
            wave_rt.drop(columns=["date"])
        ).join(
            mv.drop(columns=["date"])
        )
        out["price_volume_divergence"] = div.to_numpy()
        out["wave3_tidal_burst"] = signal.to_numpy()
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(cfg.processed_dir / f"{sid}_features.csv", index=False)

    if not all_passed:
        print("⚠ 部分股票 VG-1 未通過，結果不可視為已驗證，請依警告人工確認。")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
