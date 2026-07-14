# -*- coding: utf-8 -*-
"""pytest 共用 fixture：合成價量資料（固定種子，可重現）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from config.config import Config


@pytest.fixture()
def cfg(tmp_path):
    """測試用 Config：路徑導向 tmp、關閉抓取延遲以加速。"""
    return Config(
        cache_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        failure_log_path=tmp_path / "logs/fetch_failures.log",
        request_delay_sec_min=0.0,
        request_delay_sec_max=0.0,
        retry_backoff_base_sec=0.0,
    )


def make_synthetic_ohlcv(n=250, seed=42, stock_id="TEST"):
    """合成日線：三段趨勢（漲-跌-漲）+ 噪音，足以產生多個 ZigZag 轉折。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = np.concatenate([
        np.linspace(100, 150, n // 3),
        np.linspace(150, 118, n // 3),
        np.linspace(118, 170, n - 2 * (n // 3)),
    ])
    close = trend + rng.normal(0, 1.2, n).cumsum() * 0.3
    close = np.maximum(close, 1.0)
    spread = np.abs(rng.normal(0.8, 0.3, n))
    high = close + spread
    low = np.maximum(close - spread, 0.5)
    open_ = close + rng.normal(0, 0.4, n)
    vol = (rng.integers(8_000_000, 12_000_000, n)
           + (np.gradient(trend) > 0) * rng.integers(0, 6_000_000, n))
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "stock_id": stock_id, "date": dates,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol.astype("int64"),
    })


@pytest.fixture()
def ohlcv():
    return make_synthetic_ohlcv()
