# -*- coding: utf-8 -*-
"""auto_fetcher 測試（全 mock，不連外網）。"""
from datetime import date
from unittest import mock

import pytest

from src.fetch.auto_fetcher import (
    FugleSource, TwseSource, FallbackChain, build_default_chain,
    ApiKeyInvalidError, PlanLimitError, AllSourcesFailedError, YuantaSparkSource,
)


def _resp(status=200, payload=None):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.raise_for_status = mock.Mock()
    return r


# ---- Fugle：401 / 403 語意（LESSONLEARNT 固化為測試）----

def test_fugle_401_raises_key_invalid(cfg):
    sess = mock.Mock(); sess.get.return_value = _resp(401)
    src = FugleSource(cfg, session=sess, api_key="k")
    with pytest.raises(ApiKeyInvalidError):
        src.quote("2330")


def test_fugle_403_is_plan_limit_not_key_error(cfg):
    sess = mock.Mock(); sess.get.return_value = _resp(403)
    src = FugleSource(cfg, session=sess, api_key="k")
    with pytest.raises(PlanLimitError):
        src.quote("2330")


def test_fugle_daily_parses_candles(cfg):
    payload = {"data": [
        {"date": "2026-07-09", "open": 1000, "high": 1010,
         "low": 995, "close": 1005, "volume": 30000000},
        {"date": "2026-07-08", "open": 990, "high": 1002,
         "low": 985, "close": 1000, "volume": 25000000},
    ]}
    sess = mock.Mock(); sess.get.return_value = _resp(200, payload)
    src = FugleSource(cfg, session=sess, api_key="k")
    df = src.daily("2330", date(2026, 7, 8), date(2026, 7, 9))
    assert list(df["close"]) == [1000.0, 1005.0]        # 升冪排序
    assert df.attrs["data_source"].startswith("Fugle")
    # 驗證 key 走標頭而非 URL 參數（不得將憑證放入 URL）
    _, kwargs = sess.get.call_args
    assert kwargs["headers"]["X-API-KEY"] == "k"


def test_fugle_key_missing_env_gives_guidance(cfg, monkeypatch):
    monkeypatch.delenv("FUGLE_API_KEY", raising=False)
    with pytest.raises(ApiKeyInvalidError, match="FUGLE_API_KEY"):
        FugleSource(cfg)


# ---- TWSE MIS 快照 ----

MIS = {"msgArray": [
    {"c": "9999", "z": "1.0", "y": "1.0", "tlong": "0"},
    {"c": "2330", "z": "1005.0", "y": "1000.0", "tlong": "1751980000000"},
]}


def test_twse_quote_matches_by_item_c_not_index(cfg):
    sess = mock.Mock(); sess.get.return_value = _resp(200, MIS)
    q = TwseSource(cfg, session=sess).quote("2330")
    assert q.price == 1005.0 and q.prev_close == 1000.0   # 取第2筆，非 index 0


def test_twse_quote_after_close_z_dash(cfg):
    payload = {"msgArray": [{"c": "2330", "z": "-", "y": "1000.0", "tlong": "0"}]}
    sess = mock.Mock(); sess.get.return_value = _resp(200, payload)
    q = TwseSource(cfg, session=sess).quote("2330")
    assert q.price is None and q.prev_close == 1000.0     # 不以昨收冒充現價
    assert "非交易時段" in q.note


def test_twse_otc_prefix(cfg):
    sess = mock.Mock()
    sess.get.return_value = _resp(200, {"msgArray": [
        {"c": "6488", "z": "500.0", "y": "490.0", "tlong": "0"}]})
    TwseSource(cfg, session=sess).quote("6488.TWO")
    _, kwargs = sess.get.call_args
    assert kwargs["params"]["ex_ch"] == "otc_6488.tw"


# ---- 降級鏈 ----

def _fail_src(name, exc):
    s = mock.Mock(); s.name = name
    s.quote.side_effect = exc
    return s


def test_chain_falls_back_on_plan_limit(cfg):
    ok = mock.Mock(); ok.name = "TWSE"
    ok.quote.return_value = "OK"
    chain = FallbackChain([_fail_src("Fugle", PlanLimitError("403")), ok])
    assert chain.quote("2330") == "OK"


def test_chain_stops_on_invalid_key(cfg):
    ok = mock.Mock(); ok.name = "TWSE"
    chain = FallbackChain([_fail_src("Fugle", ApiKeyInvalidError("401")), ok])
    with pytest.raises(ApiKeyInvalidError):
        chain.quote("2330")
    ok.quote.assert_not_called()          # key 無效不得靜默降級


def test_chain_all_failed_lists_reasons(cfg):
    chain = FallbackChain([
        _fail_src("Fugle", PlanLimitError("403")),
        _fail_src("TWSE", ConnectionError("timeout")),
    ])
    with pytest.raises(AllSourcesFailedError, match="Fugle.*TWSE"):
        chain.quote("2330")


def test_default_chain_without_key_is_twse_only(cfg, monkeypatch):
    monkeypatch.delenv("FUGLE_API_KEY", raising=False)
    chain = build_default_chain(cfg)
    assert [s.name for s in chain._sources] == ["TWSE"]


def test_yuanta_spark_slot_refuses_until_approved(cfg):
    with pytest.raises(NotImplementedError, match="風險預告書"):
        YuantaSparkSource(cfg)
