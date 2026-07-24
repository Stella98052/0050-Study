# -*- coding: utf-8 -*-
"""每日收盤後更新（規格第三階段①②③）：沿用 phase1 抓取/快取/重試 +
VG-1 檢查 → 更新 realtime 特徵 → 載入 phase2 模型（特徵欄校驗）→
輸出未來 N 日預測 → 前瞻紀錄 append（Model v2 最終閘資料）。

用法：python daily_update.py --holdings holdings.csv
排程（規格④：說明即可，不實作排程器）：
  Windows 工作排程器每交易日 15:30 執行：
    schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "0050DailyUpdate" ^
      /TR "cmd /c cd /d C:\\path\\to\\phase1 && python daily_update.py --holdings holdings.csv" /ST 15:30
  （TWSE 盤後資料約 15:00 後齊備，早於此時間會抓到前一交易日）
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from config.phase2_config import Phase2Config, PHASE2_VERSION
from config.phase3_config import Phase3Config, PHASE3_VERSION
from src.fetch.holdings import load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.features.feature_matrix import build_feature_matrix
from src.validate.trading_calendar import build_union_calendar
from src.validate.vg1 import run_vg1_validation
from src.dashboard.model_service import load_model_pack, predict_latest
from src.dashboard.predictions_log import append_prediction, prospective_progress


def main() -> int:
    print(f"▶ daily_update（phase3 v{PHASE3_VERSION} / phase2 v{PHASE2_VERSION} "
          f"/ phase1 v{PHASE1_VERSION}）")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    args = ap.parse_args()
    if not args.holdings.exists():
        print(f"[中止] 找不到 {args.holdings.resolve()}")
        return 1

    p1, p2, p3 = Config(), Phase2Config(), Phase3Config()
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    snap = load_holdings_from_csv(args.holdings, end)

    frames: dict[str, pd.DataFrame] = {}
    for sid in snap.stock_ids:
        frames[sid] = fetch_stock_history(sid, start, end, p1)
    calendar = build_union_calendar(frames)

    model, bundle = load_model_pack(p3.model_path)
    if model is None:
        # L56：無模型＝當日一列預測都不會寫，導致 workflow 全綠卻無資料。
        # 此為致命狀態，必須明確失敗而非靜默略過。
        print(f"[中止] 模型包不存在（{p3.model_path.resolve()}）——"
              f"本次不會產生任何前瞻紀錄。")
        print("  診斷：data/models/*.joblib 受 .gitignore 排除，需由 "
              "workflow 的「若無模型包則先建置」步驟產生；"
              "該步驟失敗時請檢視其 log（run_phase2.py 輸出）。")
        print(f"  現存 data/models 內容："
              f"{sorted(p.name for p in Path('data/models').glob('*')) if Path('data/models').exists() else '（目錄不存在）'}")
        return 2

    rows = []
    for sid, df in frames.items():
        rpt = run_vg1_validation(df, sid, calendar, p1)
        if not rpt.passed:
            print(f"  ⚠ {sid} VG-1 未通過：{rpt.warnings[:2]}…（詳見驗證報告）")
        pred = None
        if model is not None:
            feats = build_feature_matrix(df, p1, p2)
            pred = predict_latest(model, bundle, feats)
        last_date = str(pd.to_datetime(df["date"].iloc[-1]).date())
        close = float(df["close"].iloc[-1])
        if pred:
            logged = append_prediction({
                "run_ts": datetime.now().isoformat(timespec="seconds"),
                "stock_id": sid, "last_bar_date": pred["as_of"],
                "close": close, "proba_up": pred["proba_up"],
                "pick": pred["pick"], "forward_days": p2.forward_return_days,
                "model_tag": pred["model_tag"]}, p3.predictions_csv)
            rows.append((sid, last_date, close, pred["proba_up"],
                         "✓" if logged else "略過(同日已記)"))
        else:
            rows.append((sid, last_date, close, None, "-"))

    print(f"\n{'代碼':<6}{'末根K':<12}{'收盤':>9}{'P(漲)':>8}  紀錄")
    for sid, d_, c_, p_, tag in rows:
        print(f"{sid:<6}{d_:<12}{c_:>9.2f}"
              f"{('' if p_ is None else f'{p_:>8.3f}')}  {tag}")
    today = date.today().isoformat()
    stale = [r[0] for r in rows if r[1] != today]
    if stale:
        print(f"⚠ 資料日期非今日（{len(stale)} 檔）：官方盤後約 15:00 更新，"
              f"過早執行會取得前一交易日（LESSONLEARNT 資料新鮮度）")

    prog = prospective_progress(p3.predictions_csv, p2.forward_return_days)
    print(f"\n前瞻紀錄：{prog['n_rows']} 列｜獨立樣本 {prog['n_independent']} / "
          f"{p3.min_prospective_samples}（達標前不得對 Model v2 下結論，"
          f"預先宣告規則）")
    print("提醒：預測有效性受 VG-6 現況約束（見面板狀態卡），"
          "目前模型判別力=無，紀錄僅為前瞻協定累積。")

    # ── 自選股每日技術快照（v3.22：讀 repo 凍結清單，累積方法論檢核值）──
    try:
        from pathlib import Path as _P
        from src.custom.fetch_and_save import load_watchlist
        from src.dashboard.custom_snapshots import (append_snapshot,
                                                    build_snapshot_row)
        wl = load_watchlist(_P("custom_watchlist.csv"))
        if wl:
            from src.features.feature_matrix import build_feature_matrix
            snap_path = _P("data/custom_snapshots.csv")
            run_ts = pd.Timestamp.now().isoformat(timespec="seconds")
            n_new = 0
            print(f"\n自選股快照（{len(wl)} 檔，不含模型數字）：")
            for csid in wl:
                try:
                    cdf = fetch_stock_history(csid, start, end, p1)
                    if len(cdf) == 0:
                        print(f"  ✗ {csid}：查無官方日K（上櫃/興櫃不支援）")
                        continue
                    cf = build_feature_matrix(
                        cdf.sort_values("date").reset_index(drop=True), p1, p2)
                    row = build_snapshot_row(
                        csid, cdf.sort_values("date").iloc[-1],
                        cf.tail(1).iloc[0], run_ts)
                    ok = append_snapshot(row, snap_path)
                    n_new += int(ok)
                    print(f"  {'✓' if ok else '＝'} {csid}：{row['last_bar_date']}"
                          f" 收 {row['close']} {row['tidal']}"
                          f"{'' if ok else '（當日已有紀錄，略過）'}")
                except Exception as _exc:                  # noqa: BLE001
                    print(f"  ✗ {csid}：{type(_exc).__name__}: {_exc}")
            print(f"自選股快照新增 {n_new} 筆 → data/custom_snapshots.csv")
    except Exception as _exc:                              # noqa: BLE001
        print(f"⚠ 自選股快照段失敗（不影響前十大流程）：{_exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
