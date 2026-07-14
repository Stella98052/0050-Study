# -*- coding: utf-8 -*-
"""「第 3 浪 + 潮汐爆發」訊號與獨立訊號計數（定案 4：雙軌制）。

4.1 事件層（面板展示用）：連續 True 只算 1 個事件；中斷 ≥ 1 日後再滿足
    才算新事件。
4.2 統計驗證層（VG-3 / VG-4 唯一計數依據）：兩事件間隔需 ≥ N 天
    （N = forward_return_days，報酬窗口天數）才視為統計獨立樣本；
    間隔不足者報酬窗口重疊，合併為一筆（取最早事件為代表）。
4.3 報告需雙軌並列，並附一句差距說明。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.config import Config


def detect_wave3_tidal_burst(
    wave_labels_rt: pd.DataFrame,
    mv_features: pd.DataFrame,
    divergence: pd.Series,
    cfg: Config,
) -> pd.Series:
    """★ 逐日布林訊號（以 date 為 index）：

        wave_label_realtime == '3'
        AND is_volume_burst
        AND mv_short_direction > 0
        AND NOT mv_mid_veto_active     # 定案 2：13MV 下彎 = 絕對否決
        AND NOT price_volume_divergence
    """
    sig = (
        (wave_labels_rt["wave_label_realtime"] == "3").to_numpy()
        & mv_features["is_volume_burst"].to_numpy()
        & (mv_features["mv_short_direction"] > 0).to_numpy()
        & (~mv_features["mv_mid_veto_active"]).to_numpy()
        & (~divergence.astype(bool)).to_numpy()
    )
    return pd.Series(
        sig, index=pd.to_datetime(wave_labels_rt["date"]), name="wave3_tidal_burst"
    )


def extract_signal_events(signal: pd.Series) -> list[pd.Timestamp]:
    """4.1 事件層：回傳每個事件的「起始日」清單（連續 True 為同一事件）。"""
    events: list[pd.Timestamp] = []
    prev = False
    for ts, val in signal.items():
        if bool(val) and not prev:
            events.append(ts)
        prev = bool(val)
    return events


def select_independent_dates(
    dates: list[pd.Timestamp], n_days: int
) -> list[int]:
    """【共用篩選核心，v2.3】greedy：與上一保留日相隔（日曆日）≥ n_days
    才保留，回傳保留元素之索引。VG-4 計數與權益曲線（metrics）皆委派本函式，
    確保「獨立」定義單一事實來源，不得各自維護。輸入須已按時間升冪。"""
    if not dates:
        return []
    keep = [0]
    last = pd.Timestamp(dates[0])
    for i in range(1, len(dates)):
        d = pd.Timestamp(dates[i])
        if (d - last).days >= n_days:
            keep.append(i)
            last = d
    return keep


def independent_return_series(
    returns: pd.Series, dates, n_days: int
) -> pd.Series:
    """【v2.4 共用核心】同日交易等權合併為一筆 → select_independent_dates
    篩不重疊子集。權益曲線（metrics 路徑層）與 VG-3 統計檢定「唯一」
    合法的輸入序列——定案4 原文：VG-3/VG-4 僅採統計獨立層，
    不得餵入未篩選的交易層序列（重疊觀測=偽重複，人為壓低 p 值）。"""
    df = pd.DataFrame({"r": returns.to_numpy(),
                       "d": pd.to_datetime(pd.Series(list(dates)).to_numpy())})
    df = df.dropna()
    if len(df) == 0:
        return pd.Series(dtype="float64")
    per_day = df.groupby("d")["r"].mean().sort_index()
    idx = select_independent_dates(list(per_day.index), n_days)
    return per_day.iloc[idx]


def count_statistically_independent_signals(
    events: list[pd.Timestamp], n_days: int
) -> tuple[int, list[pd.Timestamp]]:
    """4.2 統計驗證層：委派 select_independent_dates（同一套規則）。

    回傳 (統計獨立樣本數, 代表事件日清單)。VG-3 / VG-4 一律採用此計數，
    不得使用事件層計數。
    """
    idx = select_independent_dates(events, n_days)
    kept = [events[i] for i in idx]
    return len(kept), kept


def canonical_independent_samples(
    sig_bool: pd.Series, sig_returns: pd.Series | None, n_days: int
) -> tuple[int, list[pd.Timestamp], pd.Series]:
    """【v2.6 定案4正典管線】VG-3 與 VG-4 的「唯一」合法來源（同源同數）：

    ① 逐日聚合：多股序列 groupby 日期 any() → 全域排序布林（修復
       「依股票分塊 concat 未排序 → greedy 跨股跳過」的低估 bug，L16）
    ② 事件：連續訊號日合併為一事件（extract_signal_events）
    ③ 獨立：事件間隔 ≥ n_days 日曆日（count_statistically_independent_signals）
    ④ 樣本報酬：各代表日之「當日訊號股報酬均值」（sig_returns 可為 None）

    回傳 (n_independent, 代表日清單, 樣本報酬序列)。
    注意：本管線為「統計層」（事件法，定案4原文）；權益曲線之「路徑層」
    （可執行單部位、平倉後可於同一事件內再進場）另用
    independent_return_series（日期 greedy）——兩者目的不同、各自單一來源。
    """
    daily = sig_bool.groupby(sig_bool.index).any().sort_index()
    events = extract_signal_events(daily)
    n, kept = count_statistically_independent_signals(events, n_days)
    if sig_returns is None or len(sig_returns) == 0:
        return n, kept, pd.Series(dtype="float64")
    per_day = sig_returns.groupby(sig_returns.index).mean().sort_index()
    samples = per_day.reindex(pd.DatetimeIndex(kept)).dropna()
    return n, kept, samples


def independence_divergence_report(sig_bool: pd.Series, n_days: int) -> dict:
    """【v2.8 診斷工具，源自審查建議】對同一訊號集並排兩種獨立性計算：

    event_n：事件法（逐日 any 聚合→連續日合併→事件間隔≥N）——定案4，
             適用對象=稀疏 episode 型規則訊號
    path_n ：非重疊窗口法（訊號日 greedy 間隔≥N）——權益曲線/密集策略
             統計樣本適用
    signal_day_ratio：逐日聚合後為 True 的比例——接近 1 即為
             「密集策略」，事件法必然退化（整段合併為極少事件），
             此時 event_n 不具意義，統計樣本應採 path_n。
    """
    daily = sig_bool.groupby(sig_bool.index).any().sort_index()
    sig_days = [pd.Timestamp(d) for d in daily.index[daily]]
    path_idx = select_independent_dates(sig_days, n_days)
    events = extract_signal_events(daily)
    event_n, _ = count_statistically_independent_signals(events, n_days)
    ratio = float(daily.mean()) if len(daily) else 0.0
    applicable = "event" if ratio < 0.5 else "path"
    return {
        "path_n": len(path_idx),
        "event_n": event_n,
        "signal_day_ratio": round(ratio, 4),
        "applicable_method": applicable,
        "note": ("訊號日占比高（密集策略）→ 事件法語意退化，統計樣本採 path_n"
                 if applicable == "path" else
                 "訊號稀疏（episode 型）→ 統計樣本採 event_n（定案4）"),
    }


@dataclass(frozen=True)
class SignalCountReport:
    """定案 4.3：雙軌並列報告。"""

    n_events: int                       # 事件層（面板顯示用）
    n_independent: int                  # 統計獨立樣本（VG-3/VG-4 唯一依據）
    window_days: int
    note: str


def build_signal_count_report(
    signal: pd.Series, cfg: Config
) -> SignalCountReport:
    """組裝雙軌計數報告，附差距說明文字。"""
    events = extract_signal_events(signal)
    n_ind, _kept = count_statistically_independent_signals(
        events, cfg.forward_return_days
    )
    note = (
        f"事件數 {len(events)}，統計獨立樣本數 {n_ind}"
        f"（獨立門檻 = 報酬窗口 {cfg.forward_return_days} 天）；"
        "差距反映訊號報酬窗口重疊情形。VG-3/VG-4 僅採統計獨立樣本數。"
    )
    return SignalCountReport(
        n_events=len(events),
        n_independent=n_ind,
        window_days=cfg.forward_return_days,
        note=note,
    )
