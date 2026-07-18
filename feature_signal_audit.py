# -*- coding: utf-8 -*-
"""特徵訊號稽核（Model v2 研究第一步，2026/7/14 審查採納+L13修正）。

目的：區分「特徵本身沒訊號」vs「模型沒學到」——前者調參救不了，
後者才輪到模型端處理。此為最便宜的分岔診斷。

用法：python feature_signal_audit.py --holdings holdings.csv

【三條鐵則（預先宣告）】
① 只用開發集（holdout 起始日之前），本腳本以斷言強制，holdout 一次都不碰
② 偽重複修正（L13/L20）：逐檔以間隔≥N 的獨立子樣本計算相關性——
   審查原始碼直接用全部重疊列會人為壓低 p 值、長出假顯著特徵
③ 特徵數量的多重比較：全部 p 值過 Holm 校正才算數

輸出兩套獨立診斷：
A. Point-biserial：各特徵 vs label_up（獨立子樣本，Holm 校正）
B. 逐日橫斷面 IC：各特徵對 fwd_return_net 的 Spearman 等級相關，
   每隔 N 個交易日取樣一次（降低窗口重疊自相關），對日 IC 序列做
   t 檢定。限制：橫斷面僅 10 檔，單日 IC 極噪，靠 ~數百個取樣日平均。

判讀（預先宣告）：
   A、B 皆無任何特徵過 Holm → 證實「特徵無訊號」，進入特徵重設計
   （相對化/排名化優先，新資料源其次）；調參路線關閉。
   有特徵過 Holm 但模型 AUC≈0.5 → 模型端問題（過擬合/樣本量），另處理。
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, DISCLAIMER, PHASE1_VERSION
from config.phase2_config import Phase2Config, PHASE2_VERSION
from src.fetch.holdings import load_holdings_from_csv
from src.fetch.twse_daily import fetch_stock_history
from src.features.feature_matrix import build_feature_matrix
from src.model.walk_forward import holdout_start_date
from src.signal_events import select_independent_dates
from src.validate.vg3_significance import holm_correction

NUMERIC_AUDIT_COLS = [
    "mv_short", "mv_mid", "mv_long",
    "mv_short_direction", "mv_mid_direction", "mv_long_direction",
    "mv_bias", "rsi_14", "macd", "macd_signal", "macd_hist",
    "ret_5d", "ret_20d",
]
BOOL_AUDIT_COLS = ["is_volume_burst", "mv_mid_veto_active"]
WAVE_DUMMY_PREFIX = "wave_"


def independent_subsample(dev: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """L20：逐檔取間隔≥N 的獨立列（同 select_independent_dates 規則），
    彙集為稽核樣本——修正審查原始碼的偽重複缺陷。"""
    keep_idx: list = []
    for _sid, g in dev.groupby("stock_id"):
        g = g.sort_values("date")
        idx = select_independent_dates(
            [pd.Timestamp(d) for d in g["date"]], n_days)
        keep_idx.extend(g.index[idx])
    return dev.loc[keep_idx]


def audit_point_biserial(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """診斷A：獨立子樣本上，各特徵 vs label_up 之點二系列相關 + Holm。"""
    y = sub["label_up"].astype(int)
    rows = []
    for c in cols:
        x = sub[c].astype(float)
        m = x.notna() & y.notna()
        if m.sum() < 30 or x[m].std() == 0:
            rows.append({"feature": c, "r": 0.0, "p_raw": 1.0, "n": int(m.sum()),
                         "note": "無變異/樣本不足"})
            continue
        r, p = pointbiserialr(y[m], x[m])
        rows.append({"feature": c, "r": round(float(r), 4),
                     "p_raw": float(p), "n": int(m.sum()), "note": ""})
    df = pd.DataFrame(rows)
    _, adj = holm_correction(list(df["p_raw"]))
    df["p_holm"] = adj
    df["significant_holm"] = df["p_holm"] < 0.05
    return df.sort_values("p_raw").reset_index(drop=True)


def audit_daily_ic(dev: pd.DataFrame, cols: list[str], n_days: int
                   ) -> pd.DataFrame:
    """診斷B：逐日橫斷面 Spearman IC（每隔 N 日取樣），日 IC 序列 t 檢定 + Holm。"""
    dates = sorted(dev["date"].unique())
    sampled = dates[::n_days]                       # 每隔 N 個交易日取樣
    rows = []
    for c in cols:
        ics = []
        for d in sampled:
            day = dev[dev["date"] == d]
            x = day[c].astype(float)
            r_ = day["fwd_return_net"].astype(float)
            m = x.notna() & r_.notna()
            if m.sum() >= 5 and x[m].std() > 0 and r_[m].std() > 0:
                ic, _ = spearmanr(x[m], r_[m])
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) < 30:
            rows.append({"feature": c, "mean_ic": None, "t_p_raw": 1.0,
                         "n_days_sampled": len(ics)})
            continue
        ics = np.array(ics)
        t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
        from scipy.stats import t as tdist
        p = float(2 * tdist.sf(abs(t), df=len(ics) - 1))
        rows.append({"feature": c, "mean_ic": round(float(ics.mean()), 4),
                     "t_p_raw": p, "n_days_sampled": len(ics)})
    df = pd.DataFrame(rows)
    _, adj = holm_correction(list(df["t_p_raw"]))
    df["p_holm"] = adj
    df["significant_holm"] = df["p_holm"] < 0.05
    return df.sort_values("t_p_raw").reset_index(drop=True)


def audit_categorical_mw(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """診斷C（v2.11，補齊類別特徵覆蓋）：獨立子樣本上，各類別/布林特徵
    True vs False 兩組之未來淨報酬 Mann-Whitney U 檢定 + Holm。

    設計依據：提案原始碼的 median_p（逐日 p 值取中位數）非合法合併檢定量
    （無已知虛無分布），且 iloc[::N] 為逐列非逐日取樣（會砍碎橫斷面）——
    改為與診斷A同構的單一次非參數檢定，樣本獨立性由 L20 子樣本保證。"""
    from scipy.stats import mannwhitneyu
    rows = []
    # v2.13 布林驗證閘（審查採納）：值域必須 ⊆ {True,False,0,1}，
    # 否則明確跳過並印樣本值——比 to_numeric 更嚴（數值非布林亦擋）
    valid = []
    for c in cols:
        vals = sub[c].dropna()
        if vals.isin([True, False, 0, 1]).all():
            valid.append(c)
        else:
            print(f"  [跳過] {c} 非純布林欄（樣本值：{list(vals.unique()[:5])}）"
                  f"——可能誤把原始字串欄放入 cat_cols")
            rows.append({"feature": c, "n_true": 0, "n_false": 0,
                         "median_diff%": None, "p_raw": 1.0,
                         "note": "非布林欄，跳過"})
    if len(valid) < len(cols):
        print(f"  [警告] cat_cols 原有 {len(cols)} 欄，實際可檢定 {len(valid)} 欄")
    for c in valid:
        x = pd.to_numeric(sub[c], errors="coerce").astype("Int64")
        r_ = sub["fwd_return_net"].astype(float)
        m = x.notna() & r_.notna()
        pos, neg = r_[m & (x == 1)], r_[m & (x == 0)]
        if len(pos) < 30 or len(neg) < 30:
            rows.append({"feature": c, "n_true": int(len(pos)),
                         "n_false": int(len(neg)),
                         "median_diff%": None, "p_raw": 1.0,
                         "note": "某組樣本<30"})
            continue
        _, p = mannwhitneyu(pos, neg, alternative="two-sided")
        rows.append({"feature": c, "n_true": int(len(pos)),
                     "n_false": int(len(neg)),
                     "median_diff%": round((pos.median() - neg.median()) * 100, 4),
                     "p_raw": float(p), "note": ""})
    df = pd.DataFrame(rows)
    _, adj = holm_correction(list(df["p_raw"]))
    df["p_holm"] = adj
    df["significant_holm"] = df["p_holm"] < 0.05
    return df.sort_values("p_raw").reset_index(drop=True)


V2_FEATURE_COLS = [
    "ret_5d_excess", "ret_20d_excess",
    "rsi_14_rank", "mv_bias_rank", "ret_5d_rank", "ret_20d_rank",
    "realized_vol_5d", "realized_vol_20d", "vol_regime_ratio",
]


MIN_GROUP_INDEP = 200        # 分產業探索之樣本門檻（預先宣告，審查建議量級）
MIN_GROUP_STOCKS_FOR_IC = 5  # 橫斷面 IC 需組內 ≥5 檔，否則跳過並註記


def run_industry_audit(dev, sub, audit_cols, args, p1, p2, start, hstart):
    """第四優先：依官方產業分類分組重跑診斷（v2.15）。

    預先宣告規則：
    ①分組凍結 TWSE 官方分類（t187ap03_L）；不可用走 CSV；禁手寫表（L21）
    ②組內獨立樣本 < MIN_GROUP_INDEP → 整組標「樣本不足，僅供參考」，
      不納入顯著性宣告、不觸發 VG-7、不與合格組同表等重比較
    ③診斷B僅在組內股票數 ≥ MIN_GROUP_STOCKS_FOR_IC 時執行
    ④Holm 家族 = 每組內跨特徵各自一族（組間互為獨立探索）
    ⑤合格組之顯著特徵照常自動觸發 VG-7（分組樣本更小、偶然顯著
      機率更高，真偽篩反而更必要）
    """
    from src.fetch.industry_map import resolve_industry_map
    from src.fetch.twse_daily import fetch_stock_history as _fetch
    sids = sorted(dev["stock_id"].unique())
    imap, src = resolve_industry_map(sids, args.industry_csv)
    print(f"\n════ 第四優先：分產業診斷（分類來源：{src}）════")
    groups: dict[str, list[str]] = {}
    for sid in sids:
        groups.setdefault(imap[sid], []).append(sid)
    bench_df = None
    for ind, members in sorted(groups.items()):
        dev_g = dev[dev["stock_id"].isin(members)]
        sub_g = independent_subsample(dev_g, p2.forward_return_days)
        qualified = len(sub_g) >= MIN_GROUP_INDEP
        tag = "" if qualified else f"【樣本不足(<{MIN_GROUP_INDEP})，僅供參考，不納入顯著性宣告】"
        print(f"\n── 產業「{ind}」({len(members)}檔:{','.join(members)}) "
              f"獨立樣本={len(sub_g)} {tag}──")
        a_g = audit_point_biserial(sub_g, audit_cols)
        show = a_g.head(8) if qualified else a_g.head(5)
        print(show.to_string(index=False))
        if len(members) >= MIN_GROUP_STOCKS_FOR_IC:
            b_g = audit_daily_ic(dev_g, NUMERIC_AUDIT_COLS,
                                 p2.forward_return_days)
            sig_b = b_g[b_g["significant_holm"]]
            print(f"  診斷B：{'顯著=' + str(sig_b['feature'].tolist()) if len(sig_b) else '無顯著'}")
        else:
            print(f"  診斷B：組內僅 {len(members)} 檔（<{MIN_GROUP_STOCKS_FOR_IC}），"
                  f"橫斷面無意義，跳過")
        if not qualified:
            continue
        sig_feats = a_g.loc[a_g["significant_holm"], "feature"].tolist()
        if not sig_feats:
            print("  組內無特徵過 Holm。")
            continue
        from src.validate.vg7_feature_probe import probe_feature_artifact
        if bench_df is None:
            from datetime import date as _d
            bench_df = _fetch("0050", start, _d.today(), p1)
            bench_df = bench_df[pd.to_datetime(bench_df["date"]) < hstart]
        print(f"  組內顯著：{sig_feats} → 自動觸發 VG-7（分組後更需真偽篩）")
        for f_ in sig_feats:
            if f_ in dev_g.columns and pd.api.types.is_numeric_dtype(dev_g[f_]):
                rpt = probe_feature_artifact(dev_g, sub_g, bench_df, f_, p1, p2)
                print(f"    {f_}: 均值r={rpt.mean_r}(p={rpt.mean_p_holm}) "
                      f"離散r={rpt.dispersion_r}(p={rpt.dispersion_p_holm}) "
                      f"擇時p={rpt.timing_p_holm} → {rpt.verdict}")


def final_search_verdict(new_feats, old_feats, probe_reports) -> list[str]:
    """v2.15：最終搜索狀態訊息——以 VG-7 裁決為準，不得與其矛盾。

    存活 = VG-7 判「均值含資訊」或「擇時層立項」者；顯著但經 VG-7
    歸檔（假象/證據不足）者不得再印「前進折內評估」。"""
    lines: list[str] = []
    surviving = [r.feature for r in probe_reports
                 if r.verdict.startswith("均值含資訊") or "擇時層立項" in r.verdict]
    archived = [r.feature for r in probe_reports if r.feature not in surviving]
    if surviving:
        lines.append(f"→ 【VG-7 存活特徵】{sorted(surviving)}——依預先宣告條款"
                     f"停止搜尋，進入 walk-forward 折內評估（回歸+IC），"
                     f"最終 holdout 僅驗證一次。")
    if archived:
        lines.append(f"→ 【VG-7 歸檔】{sorted(archived)}——診斷A之顯著經真偽篩"
                     f"判定不可交易/證據不足；特徵搜索續行下一優先序。")
    if new_feats and not probe_reports:
        lines.append(f"→ 【新特徵含訊號（未經VG-7，非數值欄）】{new_feats}——需人工評估。")
    if old_feats and not any(r.feature in old_feats for r in probe_reports):
        lines.append(f"→ 【舊特徵有訊號但先前模型未學到】{old_feats}——"
                     f"模型端問題，另行處理。")
    if not lines:
        lines.append("→ 無存活特徵。")
    return lines


def main() -> int:
    print(f"▶ feature_signal_audit（phase2 v{PHASE2_VERSION} / phase1 v{PHASE1_VERSION}）")
    print(DISCLAIMER)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", type=Path, required=True)
    ap.add_argument("--v2", action="store_true",
                    help="加入第一/三優先新特徵（相對化/排名化/波動率）一併稽核")
    ap.add_argument("--by-industry", action="store_true",
                    help="第四優先：依 TWSE 官方產業分類分組重跑診斷")
    ap.add_argument("--industry-csv", type=Path, default=Path("industry_map.csv"),
                    help="官方分類不可用時的人工對照 CSV")
    ap.add_argument("--extra-symbols", type=str, default="",
                    help="逗號分隔之額外股票代號（方案A：擴充池特徵稽核，"
                         "僅用開發集、不碰 holdout、不重訓模型）")
    args = ap.parse_args()
    if not args.holdings.exists():
        print(f"[中止] 找不到 {args.holdings.resolve()}")
        return 1

    p1, p2 = Config(), Phase2Config()
    end = date.today()
    start = end - timedelta(days=p1.history_years * 365)
    snap = load_holdings_from_csv(args.holdings, end)
    extra = [c.strip() for c in args.extra_symbols.split(",")
             if c.strip().isdigit()]
    all_ids = list(snap.stock_ids) + [c for c in extra
                                      if c not in snap.stock_ids]
    if extra:
        print(f"方案A擴充池稽核：前十大 + 額外 {extra}"
              f"（僅開發集診斷，不碰 holdout、不重訓）")
    frames = {sid: fetch_stock_history(sid, start, end, p1)
              for sid in all_ids}
    parts = []
    for sid, df in frames.items():
        f = build_feature_matrix(df, p1, p2)
        if args.v2:
            from src.features.feature_matrix import add_volatility_features
            vol = add_volatility_features(df)
            for c in vol.columns:
                f[c] = vol[c].to_numpy()
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)
    feats["date"] = pd.to_datetime(feats["date"])
    if args.v2:
        from src.features.feature_matrix import add_market_relative_features
        print("v2 特徵：加入相對化/排名化（基準=0050 官方日K）與波動率…")
        bench = fetch_stock_history("0050", start, end, p1)
        bench_close = pd.Series(
            bench["close"].to_numpy(),
            index=pd.to_datetime(bench["date"]).dt.normalize())
        feats = add_market_relative_features(feats, bench_close)

    # 鐵則①：只用開發集，斷言強制
    all_dates = pd.DatetimeIndex(feats["date"].unique())
    hstart = pd.Timestamp(holdout_start_date(all_dates, p2))
    dev = feats[feats["date"] < hstart].copy()
    assert dev["date"].max() < hstart, "稽核觸及 holdout（違反單次使用鐵則）"
    print(f"開發集：{len(dev)} 列（< holdout 起始 {hstart.date()}），holdout 未觸及")

    # 波浪標籤 one-hot 併入稽核欄
    wave = pd.get_dummies(dev["wave_label_realtime"], prefix="wave").astype(int)
    dev = pd.concat([dev, wave], axis=1)
    audit_cols = (NUMERIC_AUDIT_COLS + BOOL_AUDIT_COLS
                  + [c for c in wave.columns])
    dev[BOOL_AUDIT_COLS] = dev[BOOL_AUDIT_COLS].astype(int)
    dev = dev.dropna(subset=["label_up", "fwd_return_net"])

    # 鐵則②：獨立子樣本
    sub = independent_subsample(dev, p2.forward_return_days)
    print(f"獨立子樣本（L13/L20 修正）：{len(sub)} 列"
          f"（自重疊列 {len(dev)} 篩得；審查原始碼若直用全列將偽重複）\n")

    print("── 診斷A：point-biserial（獨立子樣本，Holm 校正）──")
    if args.v2:
        audit_cols = audit_cols + V2_FEATURE_COLS
    a = audit_point_biserial(sub, audit_cols)
    print(a.to_string(index=False))
    print("\n── 診斷B：逐日橫斷面 IC（每隔 N 日取樣，Holm 校正）──")
    numeric_cols = NUMERIC_AUDIT_COLS + (V2_FEATURE_COLS if args.v2 else [])
    b = audit_daily_ic(dev, numeric_cols, p2.forward_return_days)
    print(b.to_string(index=False))

    print("\n── 診斷C：類別/布林特徵 Mann-Whitney（獨立子樣本，Holm 校正）──")
    cat_cols = BOOL_AUDIT_COLS + [
        c for c in sub.columns
        if c.startswith(WAVE_DUMMY_PREFIX) and c != "wave_label_realtime"]
    c_ = audit_categorical_mw(sub, cat_cols)
    print(c_.to_string(index=False))
    c_.to_csv(p2.report_dir / "feature_audit_categorical_mw.csv", index=False) \
        if p2.report_dir.exists() else None

    n_sig = (int(a["significant_holm"].sum()) + int(b["significant_holm"].sum())
             + int(c_["significant_holm"].sum()))
    print("\n===== 稽核判定（預先宣告規則）=====")
    if n_sig == 0:
        print("A、B、C 皆無特徵通過 Holm → 證實『特徵本身無訊號』（含類別特徵，"
              "覆蓋率已補齊）；"
              "調參路線關閉，進入特徵重設計（相對化/排名化優先）。")
    else:
        sig_a = a.loc[a["significant_holm"], "feature"].tolist()
        sig_b = b.loc[b["significant_holm"], "feature"].tolist()
        sig_c = c_.loc[c_["significant_holm"], "feature"].tolist()
        sig_all = set(sig_a) | set(sig_b) | set(sig_c)
        new_feats = sorted(sig_all & set(V2_FEATURE_COLS))
        old_feats = sorted(sig_all - set(V2_FEATURE_COLS))
        print(f"通過 Holm 的特徵：A={sig_a} B={sig_b} C={sig_c}")
        # v2.15：VG-7 自動觸發，最終判定訊息依 VG-7 結果收斂（消除 v2.14
        # 「上一行判歸檔、下一行喊前進」的同屏矛盾）。
        # 【Holm 家族定義（成文，審查提醒）】VG-7 內 m=3 為「每特徵各自
        # 一個確認家族」——選擇階段的多重性已由診斷 A/C 跨特徵 Holm 控制，
        # VG-7 屬對已入選者的逐一確認。若同時 ≥2 特徵觸發，仍各自 m=3，
        # 但任何子檢定 p_holm 落在 (0.01, 0.10) 邊緣帶時標「⚠ 邊緣」
        # 並要求人工複核，不得僅憑自動判定放行。
        probe_reports = []
        probe_targets = [f_ for f_ in sorted(sig_all)
                         if f_ in dev.columns
                         and pd.api.types.is_numeric_dtype(dev[f_])]
        if probe_targets:
            from src.validate.vg7_feature_probe import probe_feature_artifact
            print("\n── VG-7 特徵真偽篩（自動觸發；Holm m=3/每特徵，定義見程式註解）──")
            if len(probe_targets) > 1:
                print(f"  註：本輪 {len(probe_targets)} 個特徵同時觸發——"
                      f"各自 m=3，邊緣帶 p 一律人工複核")
            bench_df = fetch_stock_history("0050", start, end, p1)
            bench_df = bench_df[pd.to_datetime(bench_df["date"]) < hstart]
            for f_ in probe_targets:
                rpt = probe_feature_artifact(dev, sub, bench_df, f_, p1, p2)
                probe_reports.append(rpt)
                borderline = any(0.01 < pv < 0.10 for pv in
                                 (rpt.timing_p_holm, rpt.mean_p_holm,
                                  rpt.dispersion_p_holm))
                flag = ("" if rpt.is_clean_cell else "（⚠ 需人工複核）") +                        ("（⚠ 邊緣帶p，人工複核）" if borderline else "")
                print(f"  {rpt.feature}: 市場占比{rpt.market_level_share_adj:.0%} "
                      f"擇時IC={rpt.timing_ic}(p={rpt.timing_p_holm}) "
                      f"均值r={rpt.mean_r}(p={rpt.mean_p_holm}) "
                      f"離散r={rpt.dispersion_r}(p={rpt.dispersion_p_holm})")
                print(f"    → {rpt.verdict}{flag}")
        for line in final_search_verdict(new_feats, old_feats, probe_reports):
            print(line)
    if extra:
        print("\n⚠ 選股偏誤警語（預先聲明）：額外代號若因『表現好才被選入』，"
              "擴充池結果帶入選擇偏誤——任何顯著發現須先以客觀納入規則"
              "（如市值/流動性門檻於過去某日的全名單）重跑確認，才可宣告。")

    if args.by_industry:
        run_industry_audit(dev, sub, audit_cols, args, p1, p2, start, hstart)

    p2.report_dir.mkdir(parents=True, exist_ok=True)
    a.to_csv(p2.report_dir / "feature_audit_pointbiserial.csv", index=False)
    b.to_csv(p2.report_dir / "feature_audit_daily_ic.csv", index=False)
    print(f"結果已存 {p2.report_dir}/feature_audit_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
