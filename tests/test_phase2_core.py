# -*- coding: utf-8 -*-
"""第二階段核心測試：VG-5 / 標籤 / Walk-Forward / 指標 / VG-3 / VG-4。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_ohlcv
from config.phase2_config import Phase2Config
from src.features.feature_matrix import (build_feature_matrix, make_labels,
                                         report_class_balance)
from src.model.metrics import compute_backtest_metrics
from src.model.walk_forward import (WalkForwardSplit,
                                    generate_walk_forward_splits,
                                    holdout_start_date, run_walk_forward)
from src.validate.vg3_significance import bootstrap_ci, permutation_test
from src.validate.vg4_sample import build_vg4_report
from src.validate.vg5_asserts import (vg5_assert_no_retrospective,
                                      vg5_assert_train_test_no_overlap)

P2 = Phase2Config(permutation_n=200, bootstrap_n=300, n_estimators=40)


# ---------- 標籤 ----------

def test_labels_entry_next_open_exit_Nth_close(cfg):
    df = make_synthetic_ohlcv(n=30)
    lab = make_labels(df, cfg, P2)
    t = 10
    N = P2.forward_return_days
    manual = df["close"].iloc[t + N] / df["open"].iloc[t + 1] - 1
    assert abs(lab["fwd_return_gross"].iloc[t] - manual) < 1e-12
    cost = cfg.fee_buy_rate + cfg.fee_sell_rate + cfg.tax_sell_rate
    assert abs((lab["fwd_return_gross"].iloc[t]
                - lab["fwd_return_net"].iloc[t]) - cost) < 1e-12
    assert lab["fwd_return_gross"].iloc[-N:].isna().all()   # 末N列無未來窗


def test_class_balance_suggests_spw(cfg):
    s = pd.Series([True]*10 + [False]*30)
    bal = report_class_balance(s)
    assert bal["pos"] == 10 and bal["neg"] == 30
    assert abs(bal["scale_pos_weight"] - 3.0) < 1e-9


# ---------- VG-5 ----------

def test_vg5_rejects_retrospective_columns(cfg):
    df = pd.DataFrame({"a": [1], "wave_label_retrospective": ["3"]})
    with pytest.raises(AssertionError, match="retrospective"):
        vg5_assert_no_retrospective(df)


def test_vg5_truncation_recompute_passes_on_clean_features(cfg):
    """乾淨特徵：build_feature_matrix 內建截斷重算斷言不應 raise。"""
    df = make_synthetic_ohlcv(n=220)
    feats = build_feature_matrix(df, cfg, P2)          # 內部已跑 VG-5
    assert len(feats) == len(df)


def test_vg5_truncation_detects_lookahead(cfg):
    """人工植入未來函數（ret_5d 改成未來報酬）→ 截斷重算斷言必 raise。"""
    from src.validate.vg5_asserts import vg5_assert_feature_before_label
    df = make_synthetic_ohlcv(n=220)
    feats = build_feature_matrix(df, cfg, P2)
    leaked = feats.copy()
    leaked["ret_5d"] = df["close"].pct_change(5).shift(-5).to_numpy()  # 未來!
    with pytest.raises(AssertionError, match="未來"):
        vg5_assert_feature_before_label(df, leaked, cfg, P2)


def test_vg5_split_overlap_raises(cfg):
    bad = WalkForwardSplit(0, date(2020, 1, 1), date(2023, 1, 1),
                           date(2023, 1, 31), date(2023, 1, 15),  # 侵入embargo
                           date(2023, 4, 15))
    with pytest.raises(AssertionError, match="VG-5"):
        vg5_assert_train_test_no_overlap(bad)


# ---------- Walk-Forward ----------

def test_splits_respect_embargo_and_holdout(cfg):
    dates = pd.bdate_range("2016-01-01", "2026-07-01")
    splits = generate_walk_forward_splits(pd.DatetimeIndex(dates), P2)
    assert len(splits) >= 5
    h = holdout_start_date(pd.DatetimeIndex(dates), P2)
    for sp in splits:
        assert (sp.test_start - sp.train_end).days > P2.embargo_days
        assert sp.test_end < h                       # 不侵入 holdout
    # 固定輸入 → 兩次切分一致（重現性）
    splits2 = generate_walk_forward_splits(pd.DatetimeIndex(dates), P2)
    assert splits == splits2


def test_run_walk_forward_end_to_end(cfg):
    """合成十年資料端到端：至少產生一折，指標欄位齊全。"""
    df = make_synthetic_ohlcv(n=2500)                 # ~10年
    feats = build_feature_matrix(df, cfg, P2)
    folds = run_walk_forward(feats, P2)
    assert len(folds) >= 1
    m = folds[0].metrics
    assert m.n_trades >= 0 and isinstance(m.sharpe_net, float)
    assert folds[0].feature_importance                # 非黑盒


# ---------- 指標 ----------

def test_metrics_manual_example(cfg):
    net = pd.Series([0.10, -0.05, 0.02])
    dts = pd.to_datetime(["2025-01-02", "2025-01-10", "2025-01-20"])
    m = compute_backtest_metrics(net, net, dts, benchmark_return=0.03,
                                 holding_days=5)
    assert m.n_trades == 3
    assert abs(m.win_rate_net - round(2/3, 4)) < 1e-9  # 輸出4位小數
    assert abs(m.payoff_ratio_net - 1.2) < 1e-9      # 平均獲利0.06/虧0.05
    # v2.3：三筆日期相隔≥5日 → 全部不重疊 → 原始報酬直接連乘（真實路徑）
    manual_total = (1.10 * 0.95 * 1.02) - 1
    assert m.n_independent == 3
    assert abs(m.total_return_net - round(manual_total, 4)) < 1e-6
    assert abs(m.alpha_net_vs_benchmark
               - round(manual_total - 0.03, 4)) < 1e-3


def test_gross_net_gap_equals_cost(cfg):
    gross = pd.Series([0.05, 0.01, -0.02])
    cost = cfg.fee_buy_rate + cfg.fee_sell_rate + cfg.tax_sell_rate
    net = gross - cost
    dts = pd.to_datetime(["2025-01-02", "2025-01-10", "2025-01-20"])
    m = compute_backtest_metrics(gross, net, dts, 0.0, 5)
    assert m.total_return_gross > m.total_return_net    # 扣成本必較低


# ---------- VG-3 ----------

def test_vg3_noise_signal_not_significant(cfg):
    rng = np.random.default_rng(0)
    pool = pd.Series(rng.normal(0, 0.02, 3000))
    sig = pd.Series(rng.choice(pool.to_numpy(), 60))     # 純雜訊訊號
    rpt = permutation_test(sig, pool, P2)
    assert not rpt.passed
    assert "無法證明" in rpt.plain_language
    b = bootstrap_ci(sig, "mean_return", P2)
    assert (b.ci_low is None) or (b.ci_low <= 0)         # CI 應含 0 或更低


def test_vg3_true_edge_significant(cfg):
    rng = np.random.default_rng(1)
    pool = pd.Series(rng.normal(0, 0.02, 3000))
    sig = pd.Series(rng.normal(0.015, 0.02, 60))         # 真有優勢
    rpt = permutation_test(sig, pool, P2)
    assert rpt.passed and rpt.p_value < 0.05
    b = bootstrap_ci(sig, "mean_return", P2)
    assert b.passed and b.ci_low > 0


# ---------- VG-4 ----------

def test_vg4_below_threshold_marked_unreliable(cfg):
    days = pd.bdate_range("2025-01-01", periods=200)
    sig = pd.Series(False, index=days)
    sig.iloc[[10, 40, 80]] = True                        # 僅3個獨立訊號
    oos = pd.Series(False, index=pd.bdate_range("2025-10-01", periods=60))
    rpt = build_vg4_report(sig, oos, P2)
    assert rpt.n_independent == 3 and not rpt.reliable
    assert "不可靠" in rpt.statement


def test_metrics_overlapping_trades_use_daily_basket(cfg):
    """v2.1 修正鎖定：同日多筆交易先等權成日籃再累積，
    不得逐筆連乘（重疊複利虛構）。"""
    # 同一天 3 筆 +10%：日籃 = 一天 +10%，總報酬必為 0.10，非 1.1^3-1=33.1%
    net = pd.Series([0.10, 0.10, 0.10])
    same_day = pd.to_datetime(["2025-01-02"] * 3)
    m = compute_backtest_metrics(net, net, same_day, 0.0, 5)
    # v2.3：同日三筆等權合併為一筆可執行交易 → 總報酬 = +10%（非 33.1%）
    assert m.n_independent == 1
    assert abs(m.total_return_net - 0.10) < 1e-9


def test_metrics_uniform_degenerate_case_documented(cfg):
    """退化案例（連續5日全 +5%）：v2.2 與 v2.3 在此巧合同值 = 5%。
    保留此測試僅作文件化——它「無法」區分方法對錯（L10 教訓），
    真正的鎖定測試是下方的不對稱案例。v2.3 語意：僅第1筆不重疊。"""
    net = pd.Series([0.05] * 5)
    dts = pd.bdate_range("2025-01-06", periods=5)
    m = compute_backtest_metrics(net, net, dts, 0.0, 5)
    assert m.n_independent == 1
    assert abs(m.total_return_net - 0.05) < 1e-9


def test_metrics_asymmetric_returns_lock_v23(cfg):
    """【v2.3 真正的鎖定測試，源自外部審查、經獨立驗算採納】
    不對稱案例：6 個逐日訊號 [0.10,0,0,0,0,0.05]，N=5。
    正確（不重疊直接連乘）：day0 與 day5 → 1.10×1.05−1 = 15.5%。
    v2.2 日等效鏈乘會給 (1.155)^(1/5)−1 ≈ 2.92% —— 結構性錯誤。
    退化/對稱案例會讓錯誤方法蒙對答案，鎖定測試必須用不對稱案例（L10）。"""
    rets = pd.Series([0.10, 0.0, 0.0, 0.0, 0.0, 0.05])
    dts = pd.date_range("2024-01-01", periods=6, freq="D")
    m = compute_backtest_metrics(rets, rets, dts, 0.0, 5)
    assert m.n_independent == 2
    assert abs(m.total_return_net - 0.155) < 1e-9
    v22_wrong = (1.155) ** (1 / 5) - 1
    assert abs(m.total_return_net - v22_wrong) > 0.05     # 與錯誤法必須有明顯差異


def test_metrics_selection_shares_vg4_rule(cfg):
    """共用規則鎖定：metrics 的不重疊篩選與 VG-4 計數必須逐一相同
    （單一事實來源 select_independent_dates，防兩處定義漂移）。"""
    from src.signal_events import (count_statistically_independent_signals,
                                   select_independent_dates)
    dts = list(pd.to_datetime(
        ["2024-01-01", "2024-01-03", "2024-01-06", "2024-01-15", "2024-01-18"]))
    n, kept = count_statistically_independent_signals(dts, 5)
    idx = select_independent_dates(dts, 5)
    assert kept == [dts[i] for i in idx]
    rets = pd.Series([0.01] * 5)
    m = compute_backtest_metrics(rets, rets, dts, 0.0, 5)
    assert m.n_independent == n


def test_vg3_pseudo_replication_lowers_p_lock(cfg):
    """【v2.4 鎖定，L13】偽重複示範：同一份資訊（均值+0.2%）灌成 8 倍
    重複觀測，p 值被人為壓低（0.266→0.054，受控常數、固定種子）。
    這就是 VG-3 餵入未篩選重疊交易層時的失真機制——定案4 原文本就
    禁止此餵法，本測試防止回歸。"""
    p500 = Phase2Config(permutation_n=500)
    rng = np.random.default_rng(42)
    pool = pd.Series(rng.normal(0.0, 0.02, 5000))
    p_indep = permutation_test(pd.Series([0.002] * 25), pool, p500).p_value
    p_dup = permutation_test(pd.Series([0.002] * 200), pool, p500).p_value
    assert p_indep > 0.2                     # 25 筆獨立：不顯著（誠實）
    assert p_dup < p_indep * 0.5             # 偽重複：同資訊 p 被壓低過半


def test_independent_return_series_merges_and_filters(cfg):
    """共用核心：同日多筆等權合併 → ≥N 日曆日 greedy 篩選。"""
    from src.signal_events import independent_return_series
    dts = pd.to_datetime(["2024-01-01", "2024-01-01",   # 同日兩筆 → 均值
                          "2024-01-03",                  # 間隔2日 → 剔除
                          "2024-01-06"])                 # 間隔5日 → 保留
    r = pd.Series([0.10, 0.20, 0.99, 0.05])
    out = independent_return_series(r, dts, 5)
    assert len(out) == 2
    assert abs(out.iloc[0] - 0.15) < 1e-12               # (0.10+0.20)/2
    assert abs(out.iloc[1] - 0.05) < 1e-12


def test_matched_edge_series_pairs_by_entry_date(cfg):
    """v2.5 配對 edge 鎖定：逐筆扣同進場日全池均值，非全期池均。"""
    from holding_period_study import matched_edge_series
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-10")
    sig_ind = pd.Series([0.05, 0.01], index=[d1, d2])
    # 全池：d1 兩檔 (0.02, 0.04)→基準0.03；d2 兩檔 (0.02, 0.00)→基準0.01
    all_ret = pd.Series([0.02, 0.04, 0.02, 0.00], index=[d1, d1, d2, d2])
    edge = matched_edge_series(sig_ind, all_ret)
    assert abs(edge.loc[d1] - 0.02) < 1e-12        # 0.05−0.03
    assert abs(edge.loc[d2] - 0.00) < 1e-12        # 0.01−0.01
    # 全期池均=0.02 → 若誤用全期基準，d2 的 edge 會是 −0.01 而非 0


def test_vg4_multistock_concat_unsorted_fixed(cfg):
    """【v2.6 鎖定，L16】多股布林序列依股票分塊 concat（日期非全域遞增）
    餵入 VG-4，計數必須等於全域排序後的正確值——修復跨股邊界日期倒退
    導致後續股票事件被跳過的低估 bug（實跑曾同屏出現 217 vs 35 矛盾）。"""
    d = pd.bdate_range("2016-01-04", periods=8)
    a = pd.Series([True, False, False, False, False, False, True, False], index=d)
    b = pd.Series([False, False, True, False, False, False, False, True], index=d)
    concat = pd.concat([a, b])                          # 未排序（run_phase2 餵法）
    oos = pd.Series(False, index=pd.bdate_range("2025-01-01", periods=10))
    rpt = build_vg4_report(concat, oos, Phase2Config())
    # 正確：日聚合後訊號日 = 1/4,1/6(併入1/4事件? 1/6與1/4非連續→獨立事件),1/12,1/13
    # 事件起點 = 1/4, 1/6, 1/12；≥5天 greedy → 1/4, 1/12 → 2
    assert rpt.n_independent == 2


def test_vg3_sample_equals_vg4_count(cfg):
    """同源鎖定：canonical_independent_samples 的樣本數 == VG-4 計數。"""
    from src.signal_events import canonical_independent_samples
    d = pd.bdate_range("2024-01-01", periods=30)
    sig = pd.Series(False, index=d)
    sig.iloc[[0, 1, 10, 22]] = True                     # 事件: d0(含d1), d10, d22
    ret = pd.Series(0.01, index=d[[0, 1, 10, 22]])
    n, kept, samples = canonical_independent_samples(sig, ret, 5)
    oos = pd.Series(False, index=pd.bdate_range("2025-01-01", periods=5))
    rpt = build_vg4_report(sig, oos, Phase2Config())
    assert n == rpt.n_independent == 3
    assert len(samples) == n


def test_statistical_vs_path_layer_definitions_documented(cfg):
    """定義分工鎖定（L16）：連續 12 個交易日的同一事件、N=5——
    統計層（事件法，定案4）= 1 個樣本；
    路徑層（可執行再進場，日期 greedy）= 3 筆不重疊交易。
    兩者目的不同，皆為刻意設計，本測試防止任何一方被誤改成另一方。"""
    from src.signal_events import (canonical_independent_samples,
                                   independent_return_series)
    d = pd.bdate_range("2024-03-04", periods=12)
    sig = pd.Series(True, index=d)
    ret = pd.Series(0.01, index=d)
    n_stat, _, _ = canonical_independent_samples(sig, ret, 5)
    path = independent_return_series(ret, d, 5)
    assert n_stat == 1
    assert len(path) == 3


def test_holm_correction_pinned_example(cfg):
    """B1 鎖定：實跑 p 值 [0.246,0.256,0.008,0.306,0.058] 的 Holm 校正——
    N=15 調整後 0.040 仍顯著，其餘不顯著（手算驗證釘值）。"""
    from src.validate.vg3_significance import holm_correction
    reject, adj = holm_correction([0.246, 0.256, 0.008, 0.306, 0.058])
    assert adj[2] == 0.04 and reject[2] is True
    assert reject == [False, False, True, False, False]
    assert adj[0] == adj[1] == adj[3] == 0.738          # 單調化後同值
    assert adj[4] == 0.232


def test_vg3_sample_line_label_consistency(cfg):
    """A1 鎖定：VG-3 樣本行的數字必須等於 VG4Report.n_independent，
    且字串標明「統計獨立層」而非「事件法計數」（標籤曾寫反）。"""
    from run_phase2 import vg3_sample_line
    days = pd.bdate_range("2024-01-01", periods=30)
    sig = pd.Series(False, index=days)
    sig.iloc[[0, 10, 22]] = True
    oos = pd.Series(False, index=pd.bdate_range("2025-01-01", periods=5))
    rpt = build_vg4_report(sig, oos, Phase2Config())
    line = vg3_sample_line(rpt.n_independent, 99)
    assert f"獨立 {rpt.n_independent} 筆" in line
    assert "統計獨立層" in line and "事件法計數" not in line


def test_event_method_degenerates_on_dense_strategy(cfg):
    """【v2.8 鎖定，L18】密集策略（逐日聚合後幾乎天天有訊號）：
    事件法必然退化為極少事件（實跑=1），非重疊窗口法給出合理樣本數。
    診斷工具須正確判定 applicable_method='path'。"""
    from src.signal_events import (canonical_independent_samples,
                                   independence_divergence_report)
    days = pd.bdate_range("2025-07-10", periods=240)
    idx = days.repeat(2)                                # 2檔股票
    pick = pd.Series([True, False] * 240, index=idx)    # 每日恰一檔被選
    n_event, _, _ = canonical_independent_samples(pick, None, 5)
    rep = independence_divergence_report(pick, 5)
    assert n_event == 1                                 # 退化
    assert rep["event_n"] == 1
    assert rep["path_n"] >= 40                          # 240日/5 ≈ 48
    assert rep["signal_day_ratio"] == 1.0
    assert rep["applicable_method"] == "path"


def test_divergence_report_sparse_prefers_event(cfg):
    """稀疏規則訊號 → 診斷工具判定事件法適用（定案4 原始對象）。"""
    from src.signal_events import independence_divergence_report
    days = pd.bdate_range("2024-01-01", periods=200)
    sig = pd.Series(False, index=days)
    sig.iloc[[10, 11, 60, 120]] = True                  # 稀疏 episodes
    rep = independence_divergence_report(sig, 5)
    assert rep["applicable_method"] == "event"
    assert rep["event_n"] == 3                          # (10,11)併一事件+60+120


def test_vg6_flags_degenerate_outputs(cfg):
    """VG-6 鎖定：常數輸出/無判別帶/系統性單邊/AUC無技能 各自觸發。"""
    from src.validate.vg6_model_health import vg6_model_output_health
    rng = np.random.default_rng(0)
    labels = rng.random(300) > 0.5
    # 近常數：std<0.02 且全部>0.5（單邊）且擠在中間帶 → 多旗標
    r1 = vg6_model_output_health(np.full(300, 0.51), labels)
    assert not r1.passed and "near_constant_output" in r1.flags
    # 系統性偏多但有分散度：pick_rate>95% → one_sided
    p2_ = 0.6 + 0.3 * rng.random(300)
    r2 = vg6_model_output_health(p2_, labels)
    assert "one_sided_output" in r2.flags
    # 無技能：機率與標籤無關（連續均勻分布，AUC≈0.5）
    r3 = vg6_model_output_health(rng.random(2000), rng.random(2000) > 0.5)
    assert "auc_no_skill" in r3.flags


def test_vg6_passes_healthy_model(cfg):
    """健康模型：機率分散、與標籤相關（AUC 高）→ 通過。"""
    from src.validate.vg6_model_health import vg6_model_output_health
    rng = np.random.default_rng(1)
    labels = rng.random(500) > 0.5
    proba = np.clip(0.5 + (labels.astype(float) - 0.5) * 0.5
                    + rng.normal(0, 0.1, 500), 0.01, 0.99)
    rpt = vg6_model_output_health(proba, labels)
    assert rpt.passed and rpt.auc > 0.8


def test_manual_auc_pinned(cfg):
    """AUC 手算釘值：proba=[.1,.4,.35,.8], y=[0,0,1,1] → AUC=0.75。"""
    from src.validate.vg6_model_health import manual_auc
    auc = manual_auc(np.array([0.1, 0.4, 0.35, 0.8]),
                     np.array([False, False, True, True]))
    assert abs(auc - 0.75) < 1e-12


def test_trend_verdict_wording_tiers(cfg):
    """措辭分級鎖定：臨界未顯著（0.0596）與明確不顯著（0.6）語氣區隔。"""
    from holding_period_study import trend_verdict
    borderline = trend_verdict(0.09, 0.0596)
    assert "臨界未顯著" in borderline and "❌" in borderline
    clear = trend_verdict(0.09, 0.6)
    assert "臨界" not in clear and "❌" in clear
    assert "✅" in trend_verdict(0.09, 0.01)


def test_audit_detects_planted_signal_and_rejects_noise(cfg):
    """稽核鎖定：植入訊號的特徵須被抓出、純雜訊特徵過 Holm 後須不顯著。"""
    from feature_signal_audit import audit_point_biserial, independent_subsample
    rng = np.random.default_rng(7)
    n = 3000
    days = pd.bdate_range("2018-01-01", periods=n // 2)
    df = pd.DataFrame({
        "stock_id": ["A"] * (n // 2) + ["B"] * (n // 2),
        "date": list(days) * 2,
        "label_up": rng.random(n) > 0.5,
    })
    df["planted"] = df["label_up"].astype(float) * 0.8 + rng.normal(0, 0.5, n)
    for k in range(8):
        df[f"noise{k}"] = rng.normal(0, 1, n)
    sub = independent_subsample(df, 5)
    a = audit_point_biserial(sub, ["planted"] + [f"noise{k}" for k in range(8)])
    assert bool(a.loc[a["feature"] == "planted", "significant_holm"].iloc[0])
    assert not a.loc[a["feature"] != "planted", "significant_holm"].any()


def test_independent_subsample_is_per_stock_gapped(cfg):
    """L20 鎖定：稽核子樣本逐檔間隔 ≥N，跨檔不互相干擾。"""
    from feature_signal_audit import independent_subsample
    days = pd.bdate_range("2024-01-01", periods=20)
    df = pd.DataFrame({
        "stock_id": ["A"] * 20 + ["B"] * 20,
        "date": list(days) * 2,
        "label_up": [True] * 40,
    })
    sub = independent_subsample(df, 5)
    for _sid, g in sub.groupby("stock_id"):
        gaps = pd.to_datetime(g["date"]).sort_values().diff().dropna()
        assert (gaps.dt.days >= 5).all()
    assert len(sub[sub["stock_id"] == "A"]) == len(sub[sub["stock_id"] == "B"])


def test_market_relative_features(cfg):
    """相對化特徵鎖定：excess = 個股報酬 − 基準同期報酬；排名為當日百分位。"""
    from src.features.feature_matrix import add_market_relative_features
    days = pd.bdate_range("2024-02-01", periods=3)
    feats = pd.DataFrame({
        "date": list(days.repeat(2)),
        "stock_id": ["A", "B"] * 3,
        "ret_5d": [0.05, 0.01, 0.03, 0.02, 0.04, -0.01],
        "ret_20d": [0.1] * 6, "rsi_14": [70, 30, 60, 40, 55, 45],
        "mv_bias": [0.3, -0.1, 0.2, 0.0, 0.1, 0.05],
    })
    bench = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=40))
    bench[:] = np.linspace(100, 110, 40)               # 有正向漂移的基準
    out = add_market_relative_features(feats, bench)
    b5 = bench.pct_change(5).reindex(pd.to_datetime(out["date"])).to_numpy()
    assert np.allclose(out["ret_5d_excess"], out["ret_5d"] - b5)
    day0 = out[out["date"] == days[0]]
    assert day0["rsi_14_rank"].tolist() == [1.0, 0.5]  # A 排名高於 B


def test_information_coefficient_helper(cfg):
    """IC helper 鎖定：完全同序 → IC=1；打亂 → 不顯著。"""
    from src.features.feature_matrix import evaluate_information_coefficient
    rng = np.random.default_rng(3)
    y = pd.Series(rng.normal(0, 0.02, 200))
    perfect = evaluate_information_coefficient(y.to_numpy(), y)
    assert perfect["ic"] == 1.0 and perfect["has_signal"]
    shuffled = evaluate_information_coefficient(
        rng.permutation(y.to_numpy()), y)
    assert not shuffled["has_signal"]


def test_categorical_mw_detects_planted_and_rejects_noise(cfg):
    """診斷C鎖定：植入類別訊號必被抓出；八個雜訊布林過 Holm 必不顯著。
    （取代提案的 median_p——該量無虛無分布不可校準；iloc[::N] 為逐列
    非逐日取樣——兩者皆棄用，L21）"""
    from feature_signal_audit import audit_categorical_mw
    rng = np.random.default_rng(11)
    n = 2000
    ret = pd.Series(rng.normal(0, 0.02, n))
    planted = ret > ret.median()                        # 與報酬強相關的類別
    df = pd.DataFrame({"fwd_return_net": ret, "planted": planted.astype(int)})
    for k in range(8):
        df[f"nz{k}"] = (rng.random(n) > 0.5).astype(int)
    out = audit_categorical_mw(df, ["planted"] + [f"nz{k}" for k in range(8)])
    assert bool(out.loc[out["feature"] == "planted", "significant_holm"].iloc[0])
    assert not out.loc[out["feature"] != "planted", "significant_holm"].any()


def test_volatility_features_no_lookahead(cfg, ohlcv):
    """波動率特徵鎖定：rolling 手工驗算 + 截斷未來資料不改變歷史值。"""
    from src.features.feature_matrix import add_volatility_features
    v_full = add_volatility_features(ohlcv)
    t = 60
    manual = ohlcv["close"].pct_change().iloc[t-4:t+1].std()
    assert abs(v_full["realized_vol_5d"].iloc[t] - manual) < 1e-12
    v_trunc = add_volatility_features(ohlcv.iloc[:t+1].reset_index(drop=True))
    assert abs(v_full["realized_vol_5d"].iloc[t]
               - v_trunc["realized_vol_5d"].iloc[-1]) < 1e-12
    assert abs(v_full["vol_regime_ratio"].iloc[t]
               - v_trunc["vol_regime_ratio"].iloc[-1]) < 1e-12


def test_categorical_mw_skips_string_column_gracefully(cfg):
    """v2.12 鎖定：字串欄（如 wave_label_realtime 含 'unknown'）不得再
    使診斷C當機——跳過並註記，其餘欄正常檢定。"""
    from feature_signal_audit import audit_categorical_mw
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({
        "fwd_return_net": rng.normal(0, 0.02, n),
        "wave_label_realtime": ["unknown"] * n,          # 字串欄
        "flag": (rng.random(n) > 0.5).astype(int),
    })
    out = audit_categorical_mw(df, ["wave_label_realtime", "flag"])
    row = out[out["feature"] == "wave_label_realtime"].iloc[0]
    assert "跳過" in row["note"]
    assert out[out["feature"] == "flag"]["p_raw"].iloc[0] <= 1.0


def test_categorical_boolean_gate_rejects_numeric_nonbool(cfg):
    """v2.13 布林閘鎖定：數值但非布林（如 RSI）也要被擋下並註記，
    不得默默放行（to_numeric 防禦的盲點，審查補強）。"""
    from feature_signal_audit import audit_categorical_mw
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "fwd_return_net": rng.normal(0, 0.02, 200),
        "rsi_like": rng.uniform(20, 80, 200),            # 數值非布林
        "flag": (rng.random(200) > 0.5).astype(int),
    })
    out = audit_categorical_mw(df, ["rsi_like", "flag"])
    assert "非布林" in out[out["feature"] == "rsi_like"]["note"].iloc[0]


def test_vol_probe_market_level_share(cfg):
    """P1 鎖定：全池同日同值 → R²=1；完全個股獨立 → R²≈0。"""
    from vol_probe import market_level_share
    days = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(2)
    common = rng.uniform(0.01, 0.05, 100)
    df_market = pd.DataFrame({
        "date": list(days) * 3, "realized_vol_20d": list(common) * 3})
    assert market_level_share(df_market, "realized_vol_20d") == 1.0
    # 純個股雜訊：未校正 R² 機械性 ≈ 1/k=1/3，ICC 校正後應 ≈ 0
    df_idio = pd.DataFrame({
        "date": list(days) * 3,
        "realized_vol_20d": rng.uniform(0.01, 0.05, 300)})
    assert market_level_share(df_idio, "realized_vol_20d") < 0.10


def test_vol_probe_artifact_detects_dispersion_only(cfg):
    """P3 鎖定：植入「vol 只放大離散度、不移動均值」→ 離散度顯著、均值不顯著。"""
    from vol_probe import artifact_probe
    rng = np.random.default_rng(4)
    n = 2000
    vol = rng.uniform(0.01, 0.05, n)
    ret = rng.normal(0, 1, n) * vol                      # 均值0、離散∝vol
    sub = pd.DataFrame({"realized_vol_20d": vol, "fwd_return_net": ret})
    r = artifact_probe(sub)
    assert r["p_disp"] < 0.01                            # 離散度效應強
    assert r["p_mean"] > 0.05                            # 均值無效應


def test_vg7_verdict_matrix_cells(cfg):
    """VG-7 八格窮舉鎖定：假象格自動判定；「僅擇時顯著」未預期格
    必標需人工複核（is_clean_cell=False），不得自動套布林（L23）。"""
    from src.validate.vg7_feature_probe import probe_feature_artifact
    from config.phase2_config import Phase2Config
    rng = np.random.default_rng(6)
    n_days_ = 400
    days = pd.bdate_range("2018-01-01", periods=n_days_)

    def _mk(vol_to_disp: bool):
        rows = []
        for sid in ["A", "B", "C"]:
            v = rng.uniform(0.01, 0.05, n_days_)
            r = (rng.normal(0, 1, n_days_) * v if vol_to_disp
                 else rng.normal(0, 0.02, n_days_))
            rows.append(pd.DataFrame({"stock_id": sid, "date": days,
                                      "feat": v, "fwd_return_net": r}))
        return pd.concat(rows, ignore_index=True)

    bench = pd.DataFrame({
        "date": days, "open": np.linspace(100, 120, n_days_),
        "close": np.linspace(100, 120, n_days_) + rng.normal(0, 0.5, n_days_)})

    from feature_signal_audit import independent_subsample
    dev = _mk(vol_to_disp=True)
    sub = independent_subsample(dev.assign(label_up=True), 5)
    rpt = probe_feature_artifact(dev, sub, bench, "feat",
                                 Config := __import__("config.config",
                                 fromlist=["Config"]).Config(), Phase2Config())
    assert "H_artifact" in rpt.verdict and rpt.is_clean_cell

    dev2 = _mk(vol_to_disp=False)                       # 特徵與報酬完全無關
    sub2 = independent_subsample(dev2.assign(label_up=True), 5)
    rpt2 = probe_feature_artifact(dev2, sub2, bench, "feat",
                                  __import__("config.config",
                                  fromlist=["Config"]).Config(), Phase2Config())
    assert "證據不足" in rpt2.verdict


def test_industry_map_dual_key_parsing_and_floor(cfg):
    """industry_map 鎖定：中文/英文鍵名皆可解析；<100 筆視為解析失敗。"""
    from src.fetch.industry_map import fetch_industry_map

    class _Resp:
        def __init__(self, rows): self._rows = rows
        def raise_for_status(self): pass
        def json(self): return self._rows

    class _Sess:
        def __init__(self, rows): self._rows = rows
        def get(self, *a, **k): return _Resp(self._rows)

    zh = [{"公司代號": f"{1000+i}", "產業別": "半導體業"} for i in range(150)]
    m = fetch_industry_map(session=_Sess(zh))
    assert m and m["1000"] == "半導體業" and len(m) == 150
    en = [{"Code": f"{2000+i}", "industry": "Fin"} for i in range(150)]
    m2 = fetch_industry_map(session=_Sess(en))
    assert m2 and m2["2000"] == "Fin"
    assert fetch_industry_map(session=_Sess(zh[:50])) is None   # <100 → 失敗


def test_industry_map_csv_fallback_and_raise(cfg, tmp_path, monkeypatch):
    """resolve：官方不可用→CSV；兩者皆無→明確 raise（禁手寫表，L21）。"""
    import src.fetch.industry_map as im
    monkeypatch.setattr(im, "fetch_industry_map", lambda **k: None)
    csv_p = tmp_path / "industry_map.csv"
    csv_p.write_text("stock_id,industry,source\n2330,半導體,官網手抄\n",
                     encoding="utf-8")
    picked, src = im.resolve_industry_map(["2330", "9999"], csv_p)
    assert picked["2330"] == "半導體" and picked["9999"] == "CSV無分類"
    assert "manual_csv" in src
    import pytest as _pt
    with _pt.raises(RuntimeError, match="禁止手寫對照表"):
        im.resolve_industry_map(["2330"], tmp_path / "none.csv")


def test_final_search_verdict_converges_with_vg7(cfg):
    """v2.15 同屏矛盾鎖定：VG-7 判歸檔者不得再印「前進折內評估」；
    存活者才印前進（L24）。"""
    from feature_signal_audit import final_search_verdict
    from src.validate.vg7_feature_probe import VG7Report

    def _rpt(name, verdict):
        return VG7Report(feature=name, market_level_share_adj=0.2,
                         timing_ic=0.0, timing_p_holm=1.0, mean_r=0.0,
                         mean_p_holm=1.0, dispersion_r=0.3,
                         dispersion_p_holm=0.0, n_independent=500,
                         verdict=verdict, is_clean_cell=True)

    lines = final_search_verdict(
        ["realized_vol_20d"], [],
        [_rpt("realized_vol_20d", "H_artifact：…機械假象，不可交易，歸檔")])
    joined = "".join(lines)
    assert "VG-7 歸檔" in joined and "停止搜尋" not in joined

    lines2 = final_search_verdict(
        ["good_feat"], [],
        [_rpt("good_feat", "均值含資訊 → 進 walk-forward 折內回歸+IC 評估")])
    assert any("VG-7 存活特徵" in ln for ln in lines2)
