# -*- coding: utf-8 -*-
"""0050 前十大持股清單抓取。

來源白名單（依規格）：
    1. MOPS 公開資訊觀測站（mops.twse.com.tw）— ETF 申購買回清單 / 成分股揭露
    2. TWSE OpenAPI（openapi.twse.com.tw）
    3. 使用者 CSV 覆寫（manual fallback）

governance 原則：所有來源失敗且無 CSV 時 raise HoldingsUnavailableError，
錯誤訊息明示「此欄位缺失，需人工確認」。禁止回傳任何內建預設清單。

【已知限制】MOPS 頁面格式不穩定，本模組採「解析器可插拔」設計：
_SOURCE_PARSERS 依序嘗試，任一成功即回傳並記錄來源；全部失敗即 raise。
實際端點格式需於連線環境驗證（本開發容器無法連外至 twse 網域）。
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from config.config import Config
from src.schemas import HoldingsSnapshot, HoldingsUnavailableError

# MOPS ETF 成分股查詢（格式可能變動；失敗即換下一來源，不硬解析）
_MOPS_ETF_URL = "https://mops.twse.com.tw/mops/web/t78sb04"


def _try_mops(cfg: Config, session: requests.Session) -> tuple[str, ...] | None:
    """嘗試自 MOPS 取得 0050 前十大成分股。

    回傳 None 表示此來源不可用（格式變動或連線失敗），由呼叫端換下一來源。
    """
    try:
        resp = session.get(
            _MOPS_ETF_URL, timeout=cfg.request_timeout_sec,
            headers={"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0 Safari/537.36"),
                     "Accept": "application/json, text/plain, */*"})
        resp.raise_for_status()
    except Exception:                                # noqa: BLE001
        return None
    # MOPS 為表單式頁面，穩定機器可讀格式需 POST 參數且格式屢有變動；
    # 未能以可驗證方式解析時一律回傳 None（governance：不硬解析、不捏造）。
    return None


def _try_twse_openapi(cfg: Config, session: requests.Session) -> tuple[str, ...] | None:
    """嘗試自 TWSE OpenAPI 取得成分股（若對應資料集存在）。

    TWSE OpenAPI 目前未提供 0050 成分權重之標準資料集；保留此掛鉤位置，
    若日後官方新增端點，於此實作解析。現階段一律回傳 None。
    """
    return None


_SOURCE_PARSERS: tuple[tuple[str, Callable], ...] = (
    ("MOPS t78sb04", _try_mops),
    ("TWSE OpenAPI", _try_twse_openapi),
)


def fetch_0050_top10(
    cfg: Config, session: requests.Session | None = None
) -> HoldingsSnapshot:
    """依白名單來源順序抓取 0050 前十大持股。

    全部失敗 → raise HoldingsUnavailableError（不可靜默、不可捏造），
    錯誤訊息指引改用 load_holdings_from_csv 提供人工覆寫檔。
    """
    sess = session or requests.Session()
    for source_name, parser in _SOURCE_PARSERS:
        ids = parser(cfg, sess)
        if ids:
            snap = HoldingsSnapshot(
                stock_ids=tuple(ids),
                snapshot_date=date.today(),
                source=source_name,
                is_manual_override=False,
            )
            assert_holdings_count(snap, cfg)
            return snap
    raise HoldingsUnavailableError(
        "0050 前十大持股清單：所有白名單來源（MOPS / TWSE OpenAPI）皆不可用。"
        "此欄位缺失，需人工確認。請以 load_holdings_from_csv(path, snapshot_date) "
        "提供覆寫 CSV（欄位：stock_id），並註明清單基準日與出處。"
    )


def load_holdings_from_csv(path: Path, snapshot_date: date) -> HoldingsSnapshot:
    """手動 fallback：讀取使用者提供之 CSV（欄位 stock_id），標記 manual override。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = tuple(str(r["stock_id"]).strip() for r in rows if str(r["stock_id"]).strip())
    return HoldingsSnapshot(
        stock_ids=ids,
        snapshot_date=snapshot_date,
        source=f"manual_csv:{path}",
        is_manual_override=True,
    )


def assert_holdings_count(snapshot: HoldingsSnapshot, cfg: Config) -> None:
    """VG-1 子項：持股數量不等於 expected_holdings_count 時中止並提示人工確認。"""
    n = len(snapshot.stock_ids)
    if n != cfg.expected_holdings_count:
        raise ValueError(
            f"VG-1 未通過：持股清單數量為 {n}，預期 {cfg.expected_holdings_count}。"
            f"來源={snapshot.source}，需人工確認後重新提供。"
        )
