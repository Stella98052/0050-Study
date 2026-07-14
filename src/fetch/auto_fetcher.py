# -*- coding: utf-8 -*-
"""多來源自動抓取層（auto_fetcher）— TWSE / Fugle 轉接器 + 降級鏈。

設計（governance：每個回傳皆附 data_source 與 calc_logic 欄位，可追溯）：

    DataSource（協定）
    ├── TwseSource   歷史日K（STOCK_DAY 逐月）+ MIS 即時快照（輪詢）
    ├── FugleSource  歷史日K（historical/candles）+ 即時報價（intraday/quote）
    │                API key 自環境變數 FUGLE_API_KEY 讀取，絕不寫死於程式
    └── YuantaSparkSource  插槽（未實作）：需元大證券客戶臨櫃簽署風險預告書
                           並核准後，依官方 Python 元件填入；未核准前呼叫
                           一律明確 raise，不假裝可用

    FallbackChain：依序嘗試來源，全部失敗才 raise（每次失敗記錄原因）

HTTP 狀態碼語意（LESSONLEARNT 教訓，寫入程式而非僅文件）：
    401 = API key 無效 → 立即 raise（重試無意義）
    403 = 該商品不在目前方案（例：免費方案不含期貨）→ 回報 PlanLimitError，
          由降級鏈換下一來源，不可誤判為 key 錯誤
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd
import requests

# ---------------------------------------------------------------
# 共用結構
# ---------------------------------------------------------------

FUGLE_BASE = "https://api.fugle.tw/marketdata/v1.0/stock"
TWSE_MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"


class ApiKeyInvalidError(RuntimeError):
    """401：API key 無效。不重試、不降級，直接要求使用者更換 key。"""


class PlanLimitError(RuntimeError):
    """403：商品不在目前方案內。可降級至下一來源。"""


class AllSourcesFailedError(RuntimeError):
    """降級鏈全部失敗。訊息含各來源失敗原因。"""


@dataclass(frozen=True)
class Quote:
    """即時報價統一格式（含可追溯欄位）。"""

    stock_id: str
    price: float | None          # 盤中最新成交價；收盤後可能為 None（見 note）
    prev_close: float | None
    ts: str                      # 來源時間戳（原樣保留）
    data_source: str
    calc_logic: str
    note: str = ""


class DataSource(Protocol):
    """資料來源協定：任何來源都必須能報日K與即時報價。"""

    name: str

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame: ...
    def quote(self, stock_id: str) -> Quote: ...


def _get_env_key(var: str) -> str:
    """自環境變數讀取 API key。未設定 → 明確指引，不得改為寫死。"""
    key = os.environ.get(var, "").strip()
    if not key:
        raise ApiKeyInvalidError(
            f"環境變數 {var} 未設定。請於系統設定或 .env 設定後重啟，"
            "切勿將 key 寫死在程式碼或貼入對話/版本控制。"
        )
    return key


# ---------------------------------------------------------------
# TWSE 來源（歷史日K 委派 phase1 既有模組；即時 = MIS 快照輪詢）
# ---------------------------------------------------------------

class TwseSource:
    """TWSE 官方來源。免 key。

    daily：委派 phase1 的 fetch_stock_history（逐月 + 快取 + 重試）。
    quote：MIS getStockInfo 快照。上市前綴 tse_、上櫃 otc_（依 .TWO 判斷）；
           收盤後 z 為 '-'，此時回報昨收 y（LESSONLEARNT：用 item.c 對應代號，
           不可用 array index）。
    """

    name = "TWSE"

    def __init__(self, cfg, session: requests.Session | None = None):
        self._cfg = cfg
        self._sess = session or requests.Session()

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        from src.fetch.twse_daily import fetch_stock_history
        df = fetch_stock_history(stock_id, start, end, self._cfg, session=self._sess)
        df.attrs["data_source"] = "TWSE STOCK_DAY（官方，逐月）"
        df.attrs["calc_logic"] = "民國年轉西元、千分位轉數值；快取 data/raw"
        return df

    def quote(self, stock_id: str) -> Quote:
        prefix = "otc_" if stock_id.endswith(".TWO") else "tse_"
        code = stock_id.replace(".TWO", "")
        ex_ch = f"{prefix}{code}.tw"
        resp = self._sess.get(
            TWSE_MIS_URL,
            params={"ex_ch": ex_ch, "json": "1", "delay": "0"},
            headers={"Referer": "https://mis.twse.com.tw/stock/index.jsp"},
            timeout=self._cfg.request_timeout_sec,
        )
        resp.raise_for_status()
        items = resp.json().get("msgArray", [])
        item = next((it for it in items if it.get("c") == code), None)  # 用 c 對應
        if item is None:
            raise LookupError(f"MIS 回傳中找不到代號 {code}")
        z, y = item.get("z", "-"), item.get("y", "-")
        price = float(z) if z not in ("-", "", None) else None
        note = "" if price is not None else "z='-'（非交易時段），price 以 None 回報，prev_close 為昨收"
        return Quote(
            stock_id=code,
            price=price,
            prev_close=float(y) if y not in ("-", "", None) else None,
            ts=str(item.get("tlong", "")),
            data_source="TWSE MIS getStockInfo（快照輪詢）",
            calc_logic="item.c 對應代號；z=最新成交、y=昨收；z='-' 時不以昨收冒充現價",
            note=note,
        )


# ---------------------------------------------------------------
# Fugle 來源（REST；key 自 FUGLE_API_KEY 環境變數）
# ---------------------------------------------------------------

class FugleSource:
    """富果行情 REST API。

    端點（官方文件 developer.fugle.tw）：
        GET {BASE}/historical/candles/{symbol}   歷史日K
        GET {BASE}/intraday/quote/{symbol}       即時報價
    驗證：HTTP 標頭 X-API-KEY。
    狀態碼語意：401 → ApiKeyInvalidError（不重試）；403 → PlanLimitError（可降級）。
    """

    name = "Fugle"

    def __init__(self, cfg, session: requests.Session | None = None,
                 api_key: str | None = None):
        self._cfg = cfg
        self._sess = session or requests.Session()
        self._key = api_key or _get_env_key("FUGLE_API_KEY")

    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(self._cfg.max_retries):
            time.sleep(random.uniform(0.2, 0.5) if attempt else 0.0)
            resp = self._sess.get(
                url, params=params, headers={"X-API-KEY": self._key},
                timeout=self._cfg.request_timeout_sec,
            )
            if resp.status_code == 401:
                raise ApiKeyInvalidError("Fugle 回覆 401：API key 無效，請更換 key。")
            if resp.status_code == 403:
                raise PlanLimitError(
                    "Fugle 回覆 403：此商品不在目前方案內（非 key 錯誤），"
                    "降級鏈將改用下一來源。"
                )
            if resp.status_code == 429:            # 速率限制：退避後重試
                time.sleep(self._cfg.retry_backoff_base_sec * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        raise TimeoutError("Fugle 速率限制重試耗盡（429）。")

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        payload = self._get(
            f"{FUGLE_BASE}/historical/candles/{stock_id}",
            params={
                "from": start.isoformat(), "to": end.isoformat(),
                "fields": "open,high,low,close,volume",
            },
        )
        rows = payload.get("data", [])
        df = pd.DataFrame(
            {
                "stock_id": stock_id,
                "date": pd.to_datetime([r["date"] for r in rows]),
                "open": [float(r["open"]) for r in rows],
                "high": [float(r["high"]) for r in rows],
                "low": [float(r["low"]) for r in rows],
                "close": [float(r["close"]) for r in rows],
                "volume": [int(r["volume"]) for r in rows],
            }
        ).sort_values("date").reset_index(drop=True)
        df.attrs["data_source"] = "Fugle historical/candles"
        df.attrs["calc_logic"] = "官方 JSON 欄位直取，日期升冪排序"
        return df

    def quote(self, stock_id: str) -> Quote:
        d = self._get(f"{FUGLE_BASE}/intraday/quote/{stock_id}")
        return Quote(
            stock_id=stock_id,
            price=d.get("lastPrice") or d.get("closePrice"),
            prev_close=d.get("previousClose"),
            ts=str(d.get("lastUpdated", "")),
            data_source="Fugle intraday/quote",
            calc_logic="lastPrice 優先，無成交回退 closePrice；previousClose=昨收",
        )


# ---------------------------------------------------------------
# 元大 SPARK 插槽（未核准前明確不可用，不假裝）
# ---------------------------------------------------------------

class YuantaSparkSource:
    """元大 SPARK API 插槽。

    前置：需為元大證券客戶，臨櫃簽署 API 風險預告書並核准（約 5–10 個工作天），
    之後下載官方 Python 元件。核准並提供元件前，本類別一律 raise，
    不可回傳任何假資料（governance）。
    """

    name = "YuantaSPARK"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "元大 SPARK API 尚未接入：請先完成元大證券 API 申請（臨櫃簽署風險"
            "預告書，約 5–10 個工作天核准），取得官方 Python 元件後，"
            "依官方文件於此類別實作 daily/quote，登入憑證由官方元件在本機處理。"
        )


# ---------------------------------------------------------------
# 降級鏈
# ---------------------------------------------------------------

class FallbackChain:
    """依序嘗試多個來源；401 直接中止（換 key 前重試無意義），
    其餘錯誤記錄後換下一來源；全部失敗 raise 並列出各來源原因。"""

    def __init__(self, sources: list):
        self._sources = sources

    def _run(self, method: str, *args):
        errors: list[str] = []
        for src in self._sources:
            try:
                return getattr(src, method)(*args)
            except ApiKeyInvalidError:
                raise                                  # key 無效不降級
            except Exception as exc:                    # noqa: BLE001
                errors.append(f"{src.name}: {exc!r}")
        raise AllSourcesFailedError("；".join(errors))

    def daily(self, stock_id: str, start: date, end: date) -> pd.DataFrame:
        return self._run("daily", stock_id, start, end)

    def quote(self, stock_id: str) -> Quote:
        return self._run("quote", stock_id)


def build_default_chain(cfg) -> FallbackChain:
    """預設鏈：Fugle（若已設 FUGLE_API_KEY）→ TWSE。未設 key 時僅 TWSE。"""
    sources: list = []
    if os.environ.get("FUGLE_API_KEY", "").strip():
        sources.append(FugleSource(cfg))
    sources.append(TwseSource(cfg))
    return FallbackChain(sources)
