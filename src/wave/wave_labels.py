# -*- coding: utf-8 -*-
"""波浪標籤（規則化近似 approximation）與三大鐵律布林規則。

【聲明】波浪理論主觀且無法唯一驗證；本模組以轉折點序列 + 明確布林規則
推導標籤，與人工判讀可能不同。無法唯一判定 → 'unknown'，不硬判。

標籤演算法（多頭情境，approximation）：
    以「谷」為推動循環起點，交替段依序嘗試 1,2,3,4,5,A,B,C：
      - 波2 結束（谷）：檢查鐵律一（不破波1起點）；違反 → 以該谷重啟為新循環起點
      - 波3 結束（峰）：檢查鐵律二部分判斷（wave3_not_shortest_partial，
        僅比較 1 vs 3；波5 確認後才做最終判斷 wave3_not_shortest_final）
      - 波4 結束（谷）：檢查鐵律三（不與波1 價格區間重疊）；違反 → 重啟
      - 波5 之後：A（跌）、B（反彈）、C（跌），C 結束後以其谷重啟新循環
    斐波那契回撤（0.382/0.5/0.618 ± 容忍）僅作輔助標記，不作判定依據。

定案 3：realtime 中波5 未出現前，鐵律二以 `wave3_not_shortest_partial`
標示暫定通過；`wave_label_retrospective` 禁入訓練/驗證/參數選擇（VG-5）。
"""

from __future__ import annotations

import pandas as pd

from config.config import Config
from src.schemas import LabelVersion, Pivot, WaveSegment
from src.wave.zigzag import _scan_pivots, compute_pivots_retrospective

_CYCLE = ("1", "2", "3", "4", "5", "A", "B", "C")


# ---------------------------------------------------------------
# 三大鐵律（明確布林判斷式）
# ---------------------------------------------------------------

def rule_wave2_not_break_wave1_origin(w1_start_price: float, w2_end_price: float) -> bool:
    """鐵律一：第 2 浪回撤不可跌破第 1 浪起點（多頭情境）。"""
    return w2_end_price > w1_start_price


def rule_wave3_not_shortest(
    len1: float, len3: float, len5: float | None
) -> tuple[bool, str]:
    """鐵律二：第 3 浪不可為推進浪（1/3/5）中最短。

    len5=None（波5 未出現，realtime 常態）→ 僅比較 1 vs 3，
    回傳 key='wave3_not_shortest_partial'（定案 3：暫定通過標籤）；
    len5 given → 最終判斷，key='wave3_not_shortest_final'。
    """
    if len5 is None:
        return len3 >= len1, "wave3_not_shortest_partial"
    return not (len3 < len1 and len3 < len5), "wave3_not_shortest_final"


def rule_wave4_not_overlap_wave1(w1_high: float, w4_low: float) -> bool:
    """鐵律三：第 4 浪低點不可進入第 1 浪價格區間（多頭情境）。"""
    return w4_low > w1_high


# ---------------------------------------------------------------
# 推進浪 / 修正浪量化判別（approximation）
# ---------------------------------------------------------------

def classify_segment_type(
    amplitude: float,
    prev_amplitude: float | None,
    duration_bars: int,
    is_with_trend: bool,
    cfg: Config,
) -> str:
    """量化條件（不可僅憑視覺描述）：

    impulse：順主趨勢方向 AND 幅度 ≥ 相鄰逆勢段幅度 × impulse_min_amplitude_ratio
             AND 時間跨度 ≥ impulse_min_duration_bars
    corrective：逆主趨勢方向 AND 時間跨度 ≥ impulse_min_duration_bars
    其餘（資訊不足 / 條件衝突）→ 'unknown'
    """
    if duration_bars < cfg.impulse_min_duration_bars:
        return "unknown"
    if is_with_trend:
        if prev_amplitude is None:
            return "impulse"
        return (
            "impulse"
            if amplitude >= prev_amplitude * cfg.impulse_min_amplitude_ratio
            else "unknown"
        )
    return "corrective"


def _fib_hit(retrace_ratio: float, cfg: Config) -> float | None:
    """回撤比例命中之斐波那契水位（±容忍）；未命中回傳 None。僅輔助標記。"""
    for lv in cfg.fib_levels:
        if abs(retrace_ratio - lv) <= cfg.fib_tolerance:
            return lv
    return None


# ---------------------------------------------------------------
# 轉折點序列 → 波浪段標籤（核心 approximation）
# ---------------------------------------------------------------

def _label_pivot_sequence(
    pivots: list[Pivot], version: LabelVersion, cfg: Config
) -> list[WaveSegment]:
    """依演算法（見模組 docstring）將相鄰轉折點間的段落賦予波浪標籤。"""
    segments: list[WaveSegment] = []
    if len(pivots) < 2:
        return segments

    # 尋找第一個谷作為循環起點；之前的段落標 unknown
    start = 0
    while start < len(pivots) and pivots[start].kind != "trough":
        if start + 1 < len(pivots):
            segments.append(
                _make_segment(pivots[start], pivots[start + 1], "unknown",
                              "unknown", version, {}, None)
            )
        start += 1

    cyc = 0                      # 目前循環內位置（指向 _CYCLE）
    origin = pivots[start] if start < len(pivots) else None   # 波1 起點（谷）
    w1_len = w1_high = None
    lens: dict[str, float] = {}

    i = start
    while i + 1 < len(pivots):
        a, b = pivots[i], pivots[i + 1]
        amp = abs(b.price - a.price)
        dur = b.bar_index - a.bar_index
        label = _CYCLE[cyc] if cyc < len(_CYCLE) else "unknown"
        rules: dict[str, bool] = {}
        fib = None
        restart = False

        if label == "1":
            w1_len, w1_high = amp, b.price
            lens = {"1": amp}
        elif label == "2":
            ok = rule_wave2_not_break_wave1_origin(origin.price, b.price)
            rules["wave2_not_break_wave1_origin"] = ok
            if w1_len:
                fib = _fib_hit(amp / w1_len, cfg)
            if not ok:
                label, restart = "unknown", True
        elif label == "3":
            lens["3"] = amp
            ok, key = rule_wave3_not_shortest(lens.get("1", 0.0), amp, None)
            rules[key] = ok
            if not ok:
                label, restart = "unknown", True
        elif label == "4":
            ok = rule_wave4_not_overlap_wave1(w1_high, b.price)
            rules["wave4_not_overlap_wave1"] = ok
            if lens.get("3"):
                fib = _fib_hit(amp / lens["3"], cfg)
            if not ok:
                label, restart = "unknown", True
        elif label == "5":
            lens["5"] = amp
            ok, key = rule_wave3_not_shortest(
                lens.get("1", 0.0), lens.get("3", 0.0), amp
            )
            rules[key] = ok           # 鐵律二最終判斷（定案 3）
            if not ok:
                label, restart = "unknown", True

        is_with_trend = label in ("1", "3", "5") or (label in ("A", "C") and False)
        seg_type = classify_segment_type(
            amp,
            abs(a.price - pivots[i - 1].price) if i > 0 else None,
            dur,
            is_with_trend if label != "unknown" else False,
            cfg,
        ) if label != "unknown" else "unknown"

        segments.append(_make_segment(a, b, label, seg_type, version, rules, fib))

        if restart:
            # 違反鐵律：以違規段終點（若為谷）重啟為新循環起點
            if b.kind == "trough":
                origin, cyc = b, 0
            else:
                origin, cyc = None, 0
                # 峰無法作起點：往後尋谷期間標 unknown（迴圈自然處理）
        else:
            cyc += 1
            if cyc >= len(_CYCLE):    # C 浪結束 → 以其谷重啟
                origin, cyc = (b, 0) if b.kind == "trough" else (None, 0)
        if origin is None and b.kind == "trough":
            origin, cyc = b, 0
        i += 1
    return segments


def _make_segment(a, b, label, seg_type, version, rules, fib) -> WaveSegment:
    return WaveSegment(
        start_pivot=a, end_pivot=b, label=label, version=version,
        segment_type=seg_type, iron_rules=rules, fib_retracement_hit=fib,
    )


# ---------------------------------------------------------------
# 對外介面
# ---------------------------------------------------------------

def label_waves_retrospective(df: pd.DataFrame, cfg: Config) -> list[WaveSegment]:
    """◇ 回溯版標籤（version='retrospective'）。僅供視覺化，禁入訓練管線。"""
    pivots = compute_pivots_retrospective(df, cfg)
    return _label_pivot_sequence(pivots, "retrospective", cfg)


def label_waves_realtime(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """★ 即時版逐日標籤（可安全用於模型訓練）。

    實作：單次前向掃描取得全部轉折及其 confirmed_date；對每一交易日 t，
    僅採 confirmed_date <= t 的轉折重建標籤序列，當日「發展中波浪」=
    最後一段已標籤波之下一序（例：波2 谷已確認 → 當日標 '3'）。

    輸出欄位：
        date                              datetime64[ns]
        wave_label_realtime               '1'..'C' / 'unknown'
        label_basis_last_confirmed_date   當日標籤依據之最後確認轉折日
    保證逐列 label_basis_last_confirmed_date <= date（pytest 斷言）。
    """
    confirmed, _ = _scan_pivots(df, cfg.zigzag_threshold)
    dates = pd.to_datetime(df["date"]).dt.normalize()

    out_labels: list[str] = []
    out_basis: list[pd.Timestamp] = []
    k = 0                                  # 已納入之轉折數（confirmed_date <= 當日）
    cached_next = "unknown"
    cached_basis = pd.NaT

    for ts in dates:
        d = ts.date()
        advanced = False
        while k < len(confirmed) and confirmed[k].confirmed_date <= d:
            k += 1
            advanced = True
        if advanced or k == 0:
            visible = confirmed[:k]
            if visible:
                segs = _label_pivot_sequence(visible, "realtime", cfg)
                cached_next = _next_label(segs)
                cached_basis = pd.Timestamp(visible[-1].confirmed_date)
            else:
                cached_next, cached_basis = "unknown", pd.NaT
        out_labels.append(cached_next)
        out_basis.append(cached_basis)

    return pd.DataFrame(
        {
            "date": dates,
            "wave_label_realtime": out_labels,
            "label_basis_last_confirmed_date": out_basis,
        }
    )


def _next_label(segments: list[WaveSegment]) -> str:
    """由最後一段已完成波推得「當前發展中波浪」標籤。"""
    if not segments:
        # 僅有一個確認轉折、尚無完整段：若為谷則發展中為波1
        return "unknown"
    last = segments[-1]
    if last.label == "unknown":
        return "1" if last.end_pivot.kind == "trough" else "unknown"
    idx = _CYCLE.index(last.label)
    if idx + 1 < len(_CYCLE):
        return _CYCLE[idx + 1]
    return "1" if last.end_pivot.kind == "trough" else "unknown"
