# -*- coding: utf-8 -*-
"""特徵矩陣（FEATURE_SCHEMA）。★ 僅使用 realtime 波浪標籤；⛔ 內建 VG-5 斷言點。

標籤定義（定案）：
    entry_open_next  = 訊號日「下一交易日開盤價」（避免當日收盤隱性未來函數）
    fwd_return_gross = close[t+N] / open[t+1] − 1        （N = forward_return_days）
    fwd_return_net   = gross − (買費率 + 賣費率 + 證交稅) （成本近似以報酬扣減）
    label_up         = fwd_return_net > 0
末 N 列無完整未來窗 → 標籤 NaN（訓練時剔除；VG-5 以此為結構性斷言之一）。
"""

from __future__ import annotations

import pandas as pd

from src.features.tech_indicators import macd, rsi
from src.volume.volume_features import compute_mv_features
from src.wave.wave_labels import label_waves_realtime

FEATURE_COLS = [
    "wave_label_realtime",
    "mv_short", "mv_mid", "mv_long",
    "mv_short_direction", "mv_mid_direction", "mv_long_direction",
    "mv_bias", "is_volume_burst", "mv_mid_veto_active",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ret_5d", "ret_20d",
]
LABEL_COLS = ["entry_open_next", "fwd_return_gross", "fwd_return_net", "label_up"]


def _compute_features_only(ohlcv: pd.DataFrame, p1_cfg, p2_cfg) -> pd.DataFrame:
    """★ 特徵欄（不含標籤）。rolling/ewm 天然僅用截至當日資料；
    波浪標籤採 label_waves_realtime（basis_date ≤ date 逐列保證）。"""
    close = ohlcv["close"].astype("float64")
    wave = label_waves_realtime(ohlcv, p1_cfg)
    mv = compute_mv_features(ohlcv, p1_cfg)
    m, sig, hist = macd(close, p2_cfg.macd_fast, p2_cfg.macd_slow,
                        p2_cfg.macd_signal)
    out = pd.DataFrame({
        "stock_id": ohlcv["stock_id"].astype(str),
        "date": pd.to_datetime(ohlcv["date"]).dt.normalize(),
        "wave_label_realtime": wave["wave_label_realtime"].to_numpy(),
        "mv_short": mv["mv_short"].to_numpy(),
        "mv_mid": mv["mv_mid"].to_numpy(),
        "mv_long": mv["mv_long"].to_numpy(),
        "mv_short_direction": mv["mv_short_direction"].to_numpy(),
        "mv_mid_direction": mv["mv_mid_direction"].to_numpy(),
        "mv_long_direction": mv["mv_long_direction"].to_numpy(),
        "mv_bias": mv["mv_bias"].to_numpy(),
        "is_volume_burst": mv["is_volume_burst"].to_numpy(),
        "mv_mid_veto_active": mv["mv_mid_veto_active"].to_numpy(),
        "rsi_14": rsi(close, p2_cfg.rsi_window).to_numpy(),
        "macd": m.to_numpy(),
        "macd_signal": sig.to_numpy(),
        "macd_hist": hist.to_numpy(),
        "ret_5d": close.pct_change(5).to_numpy(),
        "ret_20d": close.pct_change(20).to_numpy(),
    })
    return out


def make_labels(ohlcv: pd.DataFrame, p1_cfg, p2_cfg) -> pd.DataFrame:
    """★ 標籤欄。成本 = fee_buy + fee_sell + tax_sell（來源：TWSE 現行規定）。"""
    open_ = ohlcv["open"].astype("float64")
    close = ohlcv["close"].astype("float64")
    N = p2_cfg.forward_return_days
    entry = open_.shift(-1)                       # 下一交易日開盤
    exit_ = close.shift(-N)                       # 第 N 交易日收盤
    gross = exit_ / entry - 1.0
    cost = p1_cfg.fee_buy_rate + p1_cfg.fee_sell_rate + p1_cfg.tax_sell_rate
    net = gross - cost
    return pd.DataFrame({
        "entry_open_next": entry,
        "fwd_return_gross": gross,
        "fwd_return_net": net,
        "label_up": net > 0,                      # NaN 比較為 False；訓練前以
    })                                            # gross.notna() 篩掉末 N 列


def build_feature_matrix(ohlcv: pd.DataFrame, p1_cfg, p2_cfg) -> pd.DataFrame:
    """★⛔ 單股特徵矩陣（特徵 + 標籤）。內建 VG-5 斷言：
    ① 無 retrospective 欄 ② 截斷重算一致 + 末 N 列標籤 NaN。"""
    from src.validate.vg5_asserts import (
        vg5_assert_feature_before_label, vg5_assert_no_retrospective,
    )
    feats = _compute_features_only(ohlcv, p1_cfg, p2_cfg)
    labels = make_labels(ohlcv, p1_cfg, p2_cfg)
    df = pd.concat([feats, labels], axis=1)
    vg5_assert_no_retrospective(df)
    vg5_assert_feature_before_label(ohlcv, df, p1_cfg, p2_cfg)
    return df


def report_class_balance(labels: pd.Series) -> dict:
    """★ 正負標籤分布 + scale_pos_weight 建議（負/正）。明顯失衡時印出說明。"""
    valid = labels.dropna()
    pos = int(valid.sum())
    neg = int(len(valid) - pos)
    spw = (neg / pos) if pos else float("inf")
    ratio = max(pos, neg) / max(1, min(pos, neg))
    if ratio > 1.5:
        print(f"  ⚠ 類別失衡：正={pos} 負={neg}（比 {ratio:.2f}），"
              f"建議 scale_pos_weight={spw:.3f}")
    return {"pos": pos, "neg": neg, "scale_pos_weight": round(spw, 4)}


def add_market_relative_features(
    features: pd.DataFrame, benchmark_close: pd.Series
) -> pd.DataFrame:
    """【Model v2 第二步（審查採納），稽核證實原特徵無訊號後啟用】
    絕對值特徵 → 相對值：剝離大盤/同池共同波動，讓模型看「個股相對強弱」
    而非「大盤方向」。全部僅用截至當日資料（pct_change/當日橫斷面排名）。

    新增欄位：
        ret_5d_excess / ret_20d_excess   個股過去報酬 − 0050 同期報酬
        rsi_14_rank / mv_bias_rank /
        ret_5d_rank / ret_20d_rank       同池當日百分位排名（0~1）
    """
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"])
    bench = benchmark_close.sort_index()
    b5 = bench.pct_change(5)
    b20 = bench.pct_change(20)
    out["ret_5d_excess"] = out["ret_5d"] - b5.reindex(out["date"]).to_numpy()
    out["ret_20d_excess"] = out["ret_20d"] - b20.reindex(out["date"]).to_numpy()
    for col in ("rsi_14", "mv_bias", "ret_5d", "ret_20d"):
        out[f"{col}_rank"] = out.groupby("date")[col].rank(pct=True)
    return out


def evaluate_information_coefficient(
    y_pred, y_true_return, min_abs_ic: float = 0.02
) -> dict:
    """IC 評估（回歸視角，審查採納）：Spearman 等級相關取代 AUC 評估弱訊號。
    量化慣例 |IC| > 0.02~0.05 即具實用價值。
    注意：p 值有效性同樣受樣本重疊影響——正式評估須餵獨立子樣本（L20）。"""
    from scipy.stats import spearmanr
    import numpy as _np
    yp = _np.asarray(y_pred, dtype=float)
    yt = pd.Series(y_true_return).astype(float)
    m = yt.notna().to_numpy() & ~_np.isnan(yp)
    ic, p = spearmanr(yp[m], yt[m])
    return {"ic": round(float(ic), 4), "p_value": round(float(p), 6),
            "n": int(m.sum()),
            "has_signal": bool(p < 0.05 and abs(ic) > min_abs_ic)}


def add_volatility_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """【Model v2 第三優先（提案採納）】波動率特徵——現有 23 個特徵完全
    缺席的維度。rolling 僅用截至當日資料，無未來函數。

        realized_vol_5d / realized_vol_20d   日報酬滾動標準差
        vol_regime_ratio                     5日/20日波動比（體制轉換偵測）
    """
    r = ohlcv["close"].astype("float64").pct_change()
    v5 = r.rolling(5).std()
    v20 = r.rolling(20).std()
    return pd.DataFrame({
        "realized_vol_5d": v5,
        "realized_vol_20d": v20,
        "vol_regime_ratio": v5 / v20,
    })
