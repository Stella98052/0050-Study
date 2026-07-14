# -*- coding: utf-8 -*-
"""VG-2 存活偏誤量化：對照組回測（定案：固定種子隨機抽10檔上市股，排除ETF）。

揭露警語不足以視為處理完成——對照組套用「完全相同」的特徵/訊號/模型流程。
"""
from __future__ import annotations
import random
from dataclasses import dataclass

import requests

from config.phase2_config import Phase2Config
from src.model.metrics import BacktestMetrics

_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

OPTIONAL_NOTE = ("『歷史各期實際成分股』版本為 optional/advanced 未實作；"
                 "本對照組比較是目前唯一的偏誤程度估計方式。")


@dataclass(frozen=True)
class VG2Report:
    main: BacktestMetrics
    control: BacktestMetrics
    control_mode: str
    control_stock_ids: tuple[str, ...]
    statement: str
    note_optional_version: str = OPTIONAL_NOTE


def sample_control_universe(cfg: Phase2Config, exclude: set[str] | None = None,
                            session=None, timeout: float = 25.0) -> tuple[str, ...]:
    """自 TWSE STOCK_DAY_ALL 全上市清單抽10檔（固定種子可重現）。
    排除：ETF（00開頭）、非4碼代碼、主組合持股。來源不可用→明確 raise。"""
    sess = session or requests.Session()
    last_exc = None
    for attempt in range(3):                 # v2.3：10054 連線重置屬暫時性，加重試
        try:
            resp = sess.get(_STOCK_DAY_ALL, timeout=timeout,
                            headers={"Accept": "application/json",
                                     "User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            break
        except Exception as exc:             # noqa: BLE001
            last_exc = exc
            import time as _t; _t.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"STOCK_DAY_ALL 三次重試皆失敗：{last_exc!r}")
    exclude = exclude or set()
    codes = sorted({str(r.get("Code", "")).strip() for r in resp.json()})
    pool = [c for c in codes
            if len(c) == 4 and c.isdigit() and not c.startswith("00")
            and c not in exclude]
    if len(pool) < 10:
        raise RuntimeError(f"VG-2 對照母體不足（{len(pool)} 檔），需人工確認來源")
    rng = random.Random(cfg.vg2_random_seed)
    return tuple(sorted(rng.sample(pool, 10)))


def build_vg2_report(main: BacktestMetrics, control: BacktestMetrics,
                     control_ids: tuple[str, ...], cfg: Phase2Config
                     ) -> VG2Report:
    """比較主組合 vs 對照組。main 淨報酬顯著較優（>對照 + 5個百分點，
    可解釋的粗判準）時，statement 必含存活偏誤標註，不可宣稱為策略優勢。"""
    diff = main.total_return_net - control.total_return_net
    basis = "（比較基礎：規則訊號・路徑層總報酬，主/對照同管線對稱）"
    if diff > 0.05:
        stmt = (f"主組合淨報酬高於對照組 {diff:.2%}{basis}。"
                "【必要標註】此差異可能部分或全部來自存活偏誤"
                "（以『目前』前十大持股回測歷史），而非訊號本身有效，"
                "不可直接宣稱為策略優勢。")
    elif diff < -0.05:
        stmt = f"主組合淨報酬低於對照組 {abs(diff):.2%}；未見存活偏誤造成的虛高。"
    else:
        stmt = f"主組合與對照組淨報酬差異 {diff:+.2%}，在 ±5% 內，無顯著差異。"
    return VG2Report(main=main, control=control,
                     control_mode=cfg.vg2_control_mode,
                     control_stock_ids=control_ids, statement=stmt)
