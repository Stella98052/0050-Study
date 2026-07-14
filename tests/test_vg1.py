# -*- coding: utf-8 -*-
"""VG-1 資料完整性關卡測試。"""
import pandas as pd

from conftest import make_synthetic_ohlcv
from src.validate.trading_calendar import build_union_calendar, cross_check_calendar
from src.validate.vg1 import run_vg1_validation, save_validation_report


def _calendar(df):
    return build_union_calendar({"TEST": df})


def test_clean_data_passes(cfg, ohlcv):
    rpt = run_vg1_validation(ohlcv, "TEST", _calendar(ohlcv), cfg)
    assert rpt.passed
    assert rpt.completeness_rate == 1.0


def test_detects_missing_dates(cfg, ohlcv):
    calendar = _calendar(ohlcv)                       # 完整日曆
    holed = ohlcv.drop(ohlcv.index[50:80]).reset_index(drop=True)  # 挖洞 30 日
    rpt = run_vg1_validation(holed, "TEST", calendar, cfg)
    assert not rpt.passed
    assert rpt.completeness_rate < 0.95
    assert len(rpt.missing_ranges) >= 1
    assert any("完整率" in w for w in rpt.warnings)


def test_detects_negative_volume(cfg, ohlcv):
    bad = ohlcv.copy()
    bad.loc[10, "volume"] = -5
    rpt = run_vg1_validation(bad, "TEST", _calendar(bad), cfg)
    assert not rpt.passed and rpt.n_negative_values == 1


def test_detects_duplicate_dates(cfg, ohlcv):
    bad = pd.concat([ohlcv, ohlcv.iloc[[5]]], ignore_index=True)
    rpt = run_vg1_validation(bad, "TEST", _calendar(bad), cfg)
    assert not rpt.passed and rpt.n_duplicate_dates == 1


def test_report_serialization(cfg, ohlcv, tmp_path):
    rpt = run_vg1_validation(ohlcv, "TEST", _calendar(ohlcv), cfg)
    path = save_validation_report(rpt, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "風險聲明" in text and '"passed": true' in text


def test_calendar_cross_check_flags_suspect_gap(cfg, ohlcv):
    """定案 1 輔助校驗：涵蓋範圍內的平日缺漏且非官方休市日 → 疑似缺漏警告。"""
    holed = ohlcv.drop(ohlcv.index[100:103]).reset_index(drop=True)
    calendar = build_union_calendar({"TEST": holed})
    # 官方表涵蓋整段資料期間（頭尾各放一個週六當假日，確保涵蓋範圍足夠）
    from datetime import timedelta
    p0 = ohlcv["date"].min().date(); p1 = ohlcv["date"].max().date()
    holidays = {p0 - timedelta(days=1), p1 + timedelta(days=1)}
    warns = cross_check_calendar(calendar, holidays, p0, p1)
    assert any("疑似資料缺漏而非假日" in w for w in warns)
    assert any("已知限制" in w for w in warns)          # 聯集法限制警語恆附
    assert any("涵蓋" in w for w in warns)              # 涵蓋範圍必揭露


def test_calendar_cross_check_skips_outside_coverage(cfg, ohlcv):
    """涵蓋範圍修正（2026/7/12）：休市日表僅涵蓋近年時，
    範圍外的歷史假日不得被誤報為資料缺漏。"""
    from datetime import date as _d
    holed = ohlcv.drop(ohlcv.index[100:103]).reset_index(drop=True)  # 2020年的洞
    calendar = build_union_calendar({"TEST": holed})
    holidays = {_d(2026, 1, 1), _d(2026, 2, 16)}       # 表只涵蓋 2026
    warns = cross_check_calendar(
        calendar, holidays,
        ohlcv["date"].min().date(), ohlcv["date"].max().date(),
    )
    # 2020 年的缺漏在涵蓋範圍外 → 不得出現「疑似資料缺漏」誤報
    assert not any("疑似資料缺漏" in w for w in warns)
    assert any("涵蓋" in w for w in warns)


def test_calendar_cross_check_degrades_without_holidays(cfg, ohlcv):
    calendar = _calendar(ohlcv)
    warns = cross_check_calendar(
        calendar, None,
        ohlcv["date"].min().date(), ohlcv["date"].max().date(),
    )
    assert any("輔助校驗已跳過" in w for w in warns)


def test_holiday_parser_handles_both_formats(cfg):
    """休市日解析相容 OpenAPI list-of-dicts 與官網 rwd {stat,data} 格式。"""
    from unittest import mock as _m
    from src.validate.trading_calendar import fetch_twse_holidays

    def _resp(payload):
        r = _m.Mock(); r.status_code = 200
        r.json.return_value = payload; r.raise_for_status = _m.Mock()
        return r

    # 格式①：OpenAPI（民國年）
    sess = _m.Mock()
    sess.get.return_value = _resp([{"Date": "115/01/01", "Name": "元旦"}])
    h = fetch_twse_holidays(cfg, session=sess)
    from datetime import date as _d
    assert _d(2026, 1, 1) in h

    # 格式②：官網 rwd（list-of-lists，西元）
    sess2 = _m.Mock()
    sess2.get.return_value = _resp({"stat": "OK", "data": [["2026-02-16", "春節", ""]]})
    h2 = fetch_twse_holidays(cfg, session=sess2)
    assert _d(2026, 2, 16) in h2

    # 全部失敗 → None（降級由呼叫端處理）
    sess3 = _m.Mock(); sess3.get.side_effect = ConnectionError("x")
    assert fetch_twse_holidays(cfg, session=sess3) is None
