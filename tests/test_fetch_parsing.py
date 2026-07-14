# -*- coding: utf-8 -*-
"""TWSE STOCK_DAY 解析與快取/重試測試（mock，不連外網）。"""
from datetime import date
from unittest import mock

import pandas as pd
import pytest

from src.fetch.twse_daily import (
    parse_stock_day_json, fetch_stock_month, month_range, list_failed_months,
)
from src.fetch.holdings import (
    load_holdings_from_csv, assert_holdings_count, fetch_0050_top10,
)
from src.schemas import FetchError, HoldingsUnavailableError, HoldingsSnapshot

# 官方 STOCK_DAY 回傳格式樣本（民國年、千分位逗號；欄位順序依官方 fields）
SAMPLE = {
    "stat": "OK",
    "fields": ["日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"],
    "data": [
        ["113/09/02","35,000,000","21,000,000,000","570.0","575.0","568.0","572.0","+2.0","15,000"],
        ["113/09/03","38,000,000","22,000,000,000","575.0","582.0","572.0","580.0","+8.0","18,000"],
    ],
}


def test_parse_roc_dates_and_commas():
    df = parse_stock_day_json(SAMPLE, "2330")
    assert df["date"].iloc[0] == pd.Timestamp(2024, 9, 2)   # 113+1911=2024
    assert df["volume"].iloc[1] == 38_000_000
    assert df["close"].iloc[1] == 580.0
    assert str(df["volume"].dtype) == "int64"


def test_parse_no_data_month_returns_empty():
    df = parse_stock_day_json({"stat": "查無資料"}, "2330")
    assert len(df) == 0


def test_fetch_uses_cache_without_request(cfg):
    """快取命中 → 不發任何 HTTP 請求。"""
    cpath = cfg.cache_dir / "2330" / "202409.csv"
    cpath.parent.mkdir(parents=True)
    parse_stock_day_json(SAMPLE, "2330").to_csv(cpath, index=False)
    sess = mock.Mock()
    df = fetch_stock_month("2330", "202409", cfg, session=sess)
    sess.get.assert_not_called()
    assert len(df) == 2


def test_fetch_retries_then_logs_failure(cfg):
    """連續失敗 max_retries 次 → 寫入失敗記錄檔並 raise FetchError。"""
    sess = mock.Mock()
    sess.get.side_effect = ConnectionError("boom")
    with pytest.raises(FetchError):
        fetch_stock_month("2330", "202410", cfg, session=sess)
    assert sess.get.call_count == cfg.max_retries
    assert ("2330", "202410") in list_failed_months(cfg)


def test_month_range():
    assert month_range(date(2024, 11, 15), date(2025, 2, 1)) == \
        ["202411", "202412", "202501", "202502"]


def test_holdings_csv_fallback_and_count(cfg, tmp_path):
    p = tmp_path / "holdings.csv"
    p.write_text("stock_id\n" + "\n".join(
        ["2330","2317","2454","2308","2382","2891","2881","2303","3711","2882"]
    ), encoding="utf-8")
    snap = load_holdings_from_csv(p, date(2026, 7, 1))
    assert_holdings_count(snap, cfg)               # 恰好 10 檔 → 通過
    assert snap.is_manual_override


def test_holdings_wrong_count_raises(cfg):
    snap = HoldingsSnapshot(("2330","2317"), date(2026,7,1), "manual", True)
    with pytest.raises(ValueError, match="VG-1"):
        assert_holdings_count(snap, cfg)


def test_holdings_all_sources_fail_raises(cfg):
    """所有白名單來源不可用 → 明確 raise，不可回傳內建清單（禁止捏造）。"""
    sess = mock.Mock()
    sess.get.side_effect = ConnectionError("offline")
    with pytest.raises(HoldingsUnavailableError, match="需人工確認"):
        fetch_0050_top10(cfg, session=sess)


def test_out_of_month_rows_filtered_with_warning(cfg, capsys):
    """越月列防護：快取檔含他月資料 → 過濾並印警示。"""
    from src.fetch.twse_daily import fetch_stock_month
    cpath = cfg.cache_dir / "2454" / "202409.csv"
    cpath.parent.mkdir(parents=True)
    df = parse_stock_day_json(SAMPLE, "2454")          # 2024-09 兩筆
    extra = df.copy()
    extra["date"] = pd.Timestamp(2024, 10, 1)          # 塞一筆 10 月越月列
    pd.concat([df, extra.iloc[[0]]]).to_csv(cpath, index=False)
    out = fetch_stock_month("2454", "202409", cfg, session=mock.Mock())
    assert len(out) == 2                                # 越月列被濾除
    assert (out["date"].dt.month == 9).all()
    assert "非該月資料列" in capsys.readouterr().out    # 警示可見，不靜默


def test_history_dedups_duplicate_dates_with_warning(cfg, capsys):
    """跨檔重複防護：兩個月檔含同一日期 → 合併時去重並警示。"""
    from src.fetch.twse_daily import fetch_stock_history
    d1 = cfg.cache_dir / "2308"
    d1.mkdir(parents=True)
    df_sep = parse_stock_day_json(SAMPLE, "2308")
    df_sep.to_csv(d1 / "202409.csv", index=False)
    # 10 月檔誤含 9/3 的列（模擬實跑發現的異常）+ 一筆正常 10 月列
    dup_row = df_sep.iloc[[1]].copy()
    oct_row = df_sep.iloc[[0]].copy()
    oct_row["date"] = pd.Timestamp(2024, 10, 2)
    pd.concat([oct_row, dup_row]).to_csv(d1 / "202410.csv", index=False)
    out = fetch_stock_history("2308", date(2024, 9, 1), date(2024, 10, 31), cfg,
                              session=mock.Mock())
    printed = capsys.readouterr().out
    assert not out["date"].duplicated().any()           # 最終無重複
    # 越月列已在月層被濾（10月檔的9/3列），警示可見
    assert "非該月資料列" in printed


def test_current_month_cache_is_refreshed(cfg):
    """當月快取凍結修復：進行中月份必須重抓，不得沿用舊快取。"""
    from src.fetch.twse_daily import fetch_stock_month
    # 舊快取只有 9/2 一筆（模擬月初抓的）
    cpath = cfg.cache_dir / "2330" / "202409.csv"
    cpath.parent.mkdir(parents=True)
    one_day = {"stat": "OK", "data": [SAMPLE["data"][0]]}
    parse_stock_day_json(one_day, "2330").to_csv(cpath, index=False)
    # 伺服器現在有兩筆（9/2 + 9/3）
    sess = mock.Mock()
    r = mock.Mock(); r.status_code = 200
    r.json.return_value = SAMPLE; r.raise_for_status = mock.Mock()
    sess.get.return_value = r
    # today 落在 2024-09 → 視為當月 → 必須重抓
    out = fetch_stock_month("2330", "202409", cfg, session=sess,
                            today=date(2024, 9, 20))
    sess.get.assert_called_once()
    assert len(out) == 2                               # 拿到最新兩筆
    # 快取已被覆寫為新資料
    assert len(pd.read_csv(cpath)) == 2


def test_past_month_cache_still_used(cfg):
    """歷史月份維持快取行為：不發請求。"""
    from src.fetch.twse_daily import fetch_stock_month
    cpath = cfg.cache_dir / "2330" / "202409.csv"
    cpath.parent.mkdir(parents=True)
    parse_stock_day_json(SAMPLE, "2330").to_csv(cpath, index=False)
    sess = mock.Mock()
    out = fetch_stock_month("2330", "202409", cfg, session=sess,
                            today=date(2026, 7, 12))   # 今天遠在該月之後
    sess.get.assert_not_called()
    assert len(out) == 2
