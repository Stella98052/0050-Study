# -*- coding: utf-8 -*-
"""第二階段主流程：特徵 → Walk-Forward → holdout 最終驗證 → VG-2/3/4 → 序列化。

用法（於 phase1 資料夾）：
    python run_phase2.py --holdings holdings.csv                 # 主流程
    python run_phase2.py --holdings holdings.csv --sensitivity   # 加跑敏感度(耗時)
    python run_phase2.py --holdings holdings.csv --skip-vg2      # 跳過對照組抓取

輸出：
    data/reports/phase2_report.json     完整報告（VG-1~VG-5 逐項狀態）
    data/reports/sensitivity_*.png      敏感度熱圖（--sensitivity 時）
    data/models/model_phase2-v1.joblib  序列化模型包
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from config.phase2_config import Phase2Config, PHASE2_VERSION
from src.fetch.holdings import load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.features.feature_matrix import build_feature_matrix, report_class_balance
from src.model.metrics import compute_backtest_metrics
from src.model.serialize import ModelBundle, save_bundle
from src.model.train import train_model
from src.model.walk_forward import (generate_walk_forward_splits,
                                    holdout_start_date, run_walk_forward)
from src.signal_events import detect_wave3_tidal_burst
from src.validate.vg2_survivorship import (build_vg2_report,
                                           sample_control_universe)
from src.validate.vg3_significance import bootstrap_ci, permutation_test
from src.validate.vg4_sample import build_vg4_report
from src.volume.volume_features import (compute_mv_features,
                                        detect_price_volume_divergence)
from src.wave.wave_labels import label_waves_realtime


def vg3_sample_line(n_independent: int, n_raw_trades: int) -> str:
    """A1（v2.7）：VG-3 樣本行文字。181 是「統計獨立層」計數
    （事件法『篩選後』的獨立樣本數），不是事件數本身——標籤修正並鎖測試。"""
    return (f"VG-3 檢定樣本：獨立 {n_independent} 筆"
            f"（=VG-4 統計獨立層計數，同源；交易層原始 {n_raw_trades} 筆）")


def _fetch_universe(ids, start, end, cfg) -> dict[str, pd.DataFrame]:
    out = {}
    for sid in ids:
        print(f"抓取 {sid} …")
        out[sid] = fetch_stock_history(sid, start, end, cfg)
    return out


def _benchmark_fn(cfg: Config, start: date, end: date):
    """0050 買進持有報酬函式（Alpha 對照；資料同走官方管線）。"""
    df = fetch_stock_history("0050", start, end, cfg)

    def fn(s: date, e: date) -> float:
        dts = df["date"].dt.date
        win = df[(dts >= s) & (dts <= e)]
        if len(win) < 2:
            return 0.0
        return float(win["close"].iloc[-1] / win["close"].iloc[0] - 1)
    return fn


def _strategy_signal_returns(frames, p1_cfg, p2_cfg, cutoff: date | None,
                             after: bool = False):
    """「第3浪+潮汐爆發」訊號之交易報酬序列（entry=次日開盤，N日收盤出場）。
    cutoff：after=False 取 cutoff 之前（樣本內）；True 取之後（holdout）。"""
    sig_returns, all_returns, sig_series_parts = [], [], []
    for sid, df in frames.items():
        wave = label_waves_realtime(df, p1_cfg)
        mv = compute_mv_features(df, p1_cfg)
        div = detect_price_volume_divergence(df, mv, wave)
        sig = detect_wave3_tidal_burst(wave, mv, div, p1_cfg)
        N = p2_cfg.forward_return_days
        entry = df["open"].shift(-1)
        exit_ = df["close"].shift(-N)
        gross = (exit_ / entry - 1).to_numpy()
        cost = p1_cfg.fee_buy_rate + p1_cfg.fee_sell_rate + p1_cfg.tax_sell_rate
        net = pd.Series(gross - cost, index=sig.index)
        dts = pd.Series(sig.index.date, index=sig.index)
        mask = (dts >= cutoff) if (cutoff and after) else (
            (dts < cutoff) if cutoff else pd.Series(True, index=sig.index))
        sig_returns.append(net[mask & sig.astype(bool)])
        all_returns.append(net[mask])
        sig_series_parts.append(sig[mask])
    return (pd.concat(sig_returns).dropna(), pd.concat(all_returns).dropna(),
            pd.concat(sig_series_parts))


def main() -> int:
    print(f"▶ 0050 phase2 v{PHASE2_VERSION}（phase1 v{PHASE1_VERSION}）")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--skip-vg2", action="store_true")
    args = ap.parse_args()
    if not args.holdings.exists():
        print(f"[中止] 找不到 {args.holdings.resolve()}")
        return 1

    p1, p2 = Config(), Phase2Config()
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    snap = load_holdings_from_csv(args.holdings, end)
    frames = _fetch_universe(snap.stock_ids, start, end, p1)
    bench_fn = _benchmark_fn(p1, start, end)

    # ---- 特徵（每股內建 VG-5 斷言）----
    print("建立特徵矩陣（含 VG-5 截斷重算斷言）…")
    features = pd.concat(
        [build_feature_matrix(df, p1, p2) for df in frames.values()],
        ignore_index=True).sort_values("date").reset_index(drop=True)
    vg5_pass = True                                  # 斷言未 raise 即通過

    # ---- Walk-Forward（不含 holdout）----
    all_dates = pd.DatetimeIndex(pd.to_datetime(features["date"].unique()))
    hstart = holdout_start_date(all_dates, p2)
    print(f"Walk-Forward（holdout 自 {hstart} 起完全保留）…")
    folds = run_walk_forward(features, p2, bench_fn)
    for f in folds:
        m = f.metrics
        print(f"  fold{f.split.fold_id}: 交易{m.n_trades} "
              f"Sharpe淨{m.sharpe_net} MDD{m.mdd_net} 勝率{m.win_rate_net}")

    # ---- Holdout 最終驗證（模型：全樣本內訓練 → holdout 預測）----
    dts = pd.to_datetime(features["date"]).dt.date
    valid = features["fwd_return_gross"].notna()
    tr = features[(dts < hstart) & valid]
    ho = features[(dts >= hstart) & valid]
    bal = report_class_balance(tr["label_up"])
    model, feat_names = train_model(tr, p2, bal["scale_pos_weight"])
    proba = model.predict_proba(ho)[:, 1]
    picked = ho[proba > 0.5]
    ho_metrics = compute_backtest_metrics(
        picked["fwd_return_gross"], picked["fwd_return_net"],
        dates=picked["date"],
        benchmark_return=bench_fn(hstart, end),
        holding_days=p2.forward_return_days)
    # v2.8（A2 定案）：模型策略=密集每日重選 → 事件法語意退化（曾算出
    # 統計獨立=1），統計樣本改採「非重疊窗口」（與權益曲線同一呼叫、同源），
    # 並對其跑真正的顯著性檢定，而非只靠計數閘（L18）
    from src.signal_events import (independence_divergence_report,
                                   independent_return_series)
    _pick_bool = pd.Series((proba > 0.5), index=pd.to_datetime(ho["date"]))
    _div = independence_divergence_report(_pick_bool, p2.forward_return_days)
    ind_model = independent_return_series(
        picked["fwd_return_net"], picked["date"], p2.forward_return_days)
    assert len(ind_model) == ho_metrics.n_independent, (
        "模型策略統計樣本與權益曲線路徑層必須同源（同一 independent_return_series）")
    print(f"Holdout【模型策略：LightGBM 預測>0.5，非規則訊號】：")
    print(f"  交易{ho_metrics.n_trades}｜統計樣本（非重疊窗口）"
          f"{ho_metrics.n_independent}（=權益曲線路徑層，同源）")
    print(f"  ├ 定義說明：訊號日占比 {_div['signal_day_ratio']:.0%} → 密集策略，"
          f"事件法退化（僅得 {_div['event_n']} 事件）不適用；{_div['note']}")
    print(f"  Sharpe淨{ho_metrics.sharpe_net} "
          f"總報酬淨{ho_metrics.total_return_net}（毛{ho_metrics.total_return_gross}）"
          f" Alpha{ho_metrics.alpha_net_vs_benchmark}")
    # v2.9：VG-6 模型輸出健康度（逐列機率分布 + AUC 判別力）
    from src.validate.vg6_model_health import vg6_model_output_health
    vg6 = vg6_model_output_health(proba, ho["label_up"])
    print(f"  VG-6 模型健康度：{vg6.statement}")
    if len(ind_model) < p2.min_independent_signals:
        print(f"  ⚠【VG-4 蓋過】模型策略統計樣本 {len(ind_model)} < 門檻 "
              f"{p2.min_independent_signals}，上列數字不具統計意義。")
    else:
        _boot_m = bootstrap_ci(ind_model, "mean_return", p2,
                               p2.forward_return_days)
        print(f"  模型策略顯著性（{len(ind_model)} 筆非重疊樣本）："
              f"{_boot_m.plain_language}")

    # ---- 「第3浪+潮汐爆發」訊號統計 + VG-3 / VG-4 ----
    sig_ret_is, all_ret_is, sig_is = _strategy_signal_returns(
        frames, p1, p2, hstart, after=False)
    _, _, sig_oos = _strategy_signal_returns(frames, p1, p2, hstart, after=True)
    # v2.6：VG-3 樣本取自定案4正典管線（逐日聚合→事件→≥N），與 VG-4 同源同數
    from src.signal_events import canonical_independent_samples
    n_can, _kept, sig_ind_is = canonical_independent_samples(
        sig_is, sig_ret_is, p2.forward_return_days)
    print("【規則訊號：第3浪+潮汐爆發（與上方模型策略為不同對象）】")
    print(vg3_sample_line(n_can, len(sig_ret_is)))       # A1：標籤修正+可測
    vg3_perm = permutation_test(sig_ind_is, all_ret_is, p2)
    vg3_boot = bootstrap_ci(sig_ind_is, "mean_return", p2,
                            p2.forward_return_days)
    print(f"VG-3 permutation：{vg3_perm.plain_language}")
    print(f"VG-3 bootstrap：{vg3_boot.plain_language}")
    vg4 = build_vg4_report(sig_is, sig_oos, p2)
    assert n_can == vg4.n_independent, (
        f"VG-3 樣本數({n_can}) != VG-4 統計獨立層計數({vg4.n_independent})，"
        "兩者必須源自同一正典管線")
    print(f"VG-4：{vg4.statement}")

    # ---- VG-2 對照組 ----
    vg2 = None
    if not args.skip_vg2:
        print("VG-2 對照組（隨機10檔上市股，固定種子）…")
        try:
            ctrl_ids = sample_control_universe(p2, exclude=set(snap.stock_ids))
            ctrl_frames = _fetch_universe(ctrl_ids, start, end, p1)
            ctrl_ret, _, _ = _strategy_signal_returns(
                ctrl_frames, p1, p2, hstart, after=False)
            main_metrics = compute_backtest_metrics(
                sig_ret_is, sig_ret_is, sig_ret_is.index,
                0.0, p2.forward_return_days)
            ctrl_metrics = compute_backtest_metrics(
                ctrl_ret, ctrl_ret, ctrl_ret.index,
                0.0, p2.forward_return_days)
            vg2 = build_vg2_report(main_metrics, ctrl_metrics, ctrl_ids, p2)
            print(f"VG-2：{vg2.statement}")
        except Exception as exc:                     # noqa: BLE001
            print(f"⚠ VG-2 對照組執行失敗（{exc!r}）——狀態記為未通過，不可省略")

    # ---- 敏感度（選跑，樣本內標註）----
    sens_paths = []
    if args.sensitivity:
        from src.model.sensitivity import (plot_sensitivity_heatmaps,
                                           zigzag_sensitivity_analysis)
        sens = zigzag_sensitivity_analysis(frames, p1, p2, bench_fn)
        sens_paths = [str(x) for x in
                      plot_sensitivity_heatmaps(sens, p2.report_dir)]
        print(sens.to_string(index=False))

    # ---- VG-1 狀態（讀第一階段報告）----
    vg1_pass = all(
        json.loads(p.read_text(encoding="utf-8")).get("passed", False)
        for p in Path("data").glob("validation_report_*.json")
    ) if list(Path("data").glob("validation_report_*.json")) else False

    # ---- 序列化 ----
    vg_summary = {
        "VG-1": vg1_pass, "VG-2": bool(vg2 is not None),
        "VG-3": bool(vg3_perm.passed or vg3_boot.passed),
        "VG-4": vg4.reliable, "VG-5": vg5_pass,
        "VG-6": vg6.passed,
    }
    bundle = ModelBundle(
        feature_names=tuple(feat_names),
        zigzag_threshold_used=p1.zigzag_threshold,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        version_tag=p2.bundle_version_tag, vg_summary=vg_summary,
        p2_config={k: str(v) for k, v in dataclasses.asdict(p2).items()},
    )
    mpath = save_bundle(model, bundle, p2)

    # ---- 最終報告（逐項 VG，未通過不可省略）----
    p2.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "disclaimer": DISCLAIMER,
        "phase2_version": PHASE2_VERSION,
        "holdout_start": str(hstart),
        "walk_forward_folds": [
            {"fold": f.split.fold_id, **dataclasses.asdict(f.metrics)}
            for f in folds],
        "holdout_metrics": dataclasses.asdict(ho_metrics),
        "vg3_permutation": dataclasses.asdict(vg3_perm),
        "vg3_bootstrap": dataclasses.asdict(vg3_boot),
        "vg4": dataclasses.asdict(vg4),
        "vg2": dataclasses.asdict(vg2) if vg2 else "未執行/失敗（未通過）",
        "vg6": dataclasses.asdict(vg6),
        "vg_summary": vg_summary,
        "model_bundle": str(mpath),
        "sensitivity_plots": sens_paths,
        "top_features": (folds[-1].feature_importance if folds else {}),
    }
    rpath = p2.report_dir / "phase2_report.json"
    rpath.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                default=str), encoding="utf-8")
    print("\n===== 驗證關卡總結（未通過項不可省略）=====")
    for k, v in vg_summary.items():
        print(f"  {k}: {'✅ 通過' if v else '❌ 未通過'}")
    print(f"報告：{rpath}｜模型包：{mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
