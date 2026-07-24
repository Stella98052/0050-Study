# -*- coding: utf-8 -*-
"""評分模型與籌碼資料鎖定測試（v3.27：六因子＋聖杯權重）。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from config.score_config import ScoreConfig, golden_weights
from src.score.simple_score import (compute_simple_score, score_growth,
                                    score_institution, score_margin,
                                    score_quality, score_tide,
                                    score_valuation)


def test_golden_ratio_weights():
    """聖杯比例：13:8:5:3:2:1，相鄰比值趨近 0.618，總和 100。"""
    w = golden_weights()
    assert abs(sum(w.values()) - 100.0) < 0.05
    vals = list(w.values())
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 2)]
    for r in ratios:
        assert 0.55 < r < 0.70, f"相鄰比 {r} 偏離黃金比例"
    assert vals == sorted(vals, reverse=True)              # 遞減


def test_growth_and_quality_subfactors():
    """成長/品質：子項缺一仍可算（自動平均），全缺回 None。"""
    v, t = score_growth(rev_yoy=0.40, eps_accel=0.30)
    assert abs(v - 1.0) < 1e-9 and "EPS加速度" in t
    v2, _ = score_growth(rev_yoy=0.40)                     # 只有營收
    assert v2 == 1.0
    assert score_growth()[0] is None
    q, qt = score_quality(margins={"gross_yoy": 0.03}, roe=0.20, fcf=1.0)
    assert abs(q - 1.0) < 1e-9 and "ROE" in qt
    assert score_quality(margins={}, roe=None, fcf=None)[0] is None
    # EPS 衰退（負加速度）分數應低於 0.5
    assert score_growth(eps_accel=-0.30)[0] < 0.5


def test_quality_roe_floor_and_fcf_sign():
    """ROE 低於下限得 0；FCF 為負得 0、為正得 1。"""
    assert score_quality(roe=0.05)[0] == 0.0
    assert score_quality(fcf=-1.0)[0] == 0.0
    assert score_quality(fcf=1.0)[0] == 1.0


def test_institution_streak_and_ratio():
    net = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    vol = pd.Series([10000.0] * 5)
    v, txt = score_institution(net, vol)
    assert 0.0 <= v <= 1.0 and "連買 5 日" in txt
    assert score_institution(pd.Series([-100.0] * 5), vol)[0] < v
    assert score_institution(pd.Series([1.0]), vol)[0] is None


def test_margin_and_tide_and_valuation():
    down = pd.Series([1000.0, 950.0, 900.0, 880.0, 850.0])
    up = pd.Series([850.0, 880.0, 900.0, 950.0, 1000.0])
    assert score_margin(down, pd.Series([200.0]))[0] > \
        score_margin(up, pd.Series([200.0]))[0]
    assert score_tide(1, 1, False)[0] == 1.0
    assert score_tide(1, 1, True)[0] == 0.0
    hist = pd.Series(range(10, 110))
    assert score_valuation(15.0, hist)[0] > score_valuation(105.0, hist)[0]
    assert score_valuation(15.0, pd.Series([1.0] * 10))[0] is None


def test_compute_score_redistribution_and_veto():
    """缺項只用可用權重；13MV 下彎硬性封頂且低於未否決情形。"""
    base = dict(rev_yoy=0.40, eps_accel=0.30, roe=0.20, fcf=1.0,
                margins={"gross_yoy": 0.03},
                inst_net=pd.Series([100.0] * 5),
                volume=pd.Series([1000.0] * 5),
                margin_bal=pd.Series([1000.0, 950, 900, 880, 850]),
                short_bal=pd.Series([300.0]),
                pe=10.0, pe_hist=pd.Series(range(10, 130)))
    full = compute_simple_score(**base, mv_short_dir=1, mv_mid_dir=1)
    assert full["n_available"] == 6 and full["score"] > 90
    veto = compute_simple_score(**base, mv_short_dir=1, mv_mid_dir=-1)
    assert veto["capped_by_13mv"] and veto["score"] <= 20.0
    partial = compute_simple_score(rev_yoy=0.40, mv_short_dir=1, mv_mid_dir=1)
    assert partial["n_available"] == 2
    assert set(partial["weight_used"]) == {"growth", "tide"}
    assert partial["score"] == 100.0


def test_config_thresholds_are_tunable():
    """門檻可調：改 config 即改變分數（未來校準用）。"""
    strict = ScoreConfig(rev_yoy_full=0.80)
    a = compute_simple_score(rev_yoy=0.40, mv_short_dir=0, mv_mid_dir=0)
    b = compute_simple_score(rev_yoy=0.40, mv_short_dir=0, mv_mid_dir=0,
                             cfg=strict)
    assert b["score"] < a["score"], "門檻放寬後分數應下降"


def test_every_component_reports_label_weight_logic():
    """治理：每分項須附 label／weight／calc_logic。"""
    r = compute_simple_score(rev_yoy=0.2, mv_short_dir=1, mv_mid_dir=1)
    for k, d in r["detail"].items():
        assert d["label"] and isinstance(d["calc_logic"], str)
        assert d["weight"] > 0


def test_parse_t86_field_driven():
    """T86 解析：欄名驅動（欄序不可假設）、非數字代號剔除、千分位。"""
    from src.fetch.twse_chips import parse_t86
    payload = {"fields": ["證券代號", "證券名稱", "外陸資買賣超股數(不含外資自營商)",
                          "投信買賣超股數", "三大法人買賣超股數"],
               "data": [["2330", "台積電", "1,000", "500", "2,500"],
                        ["合計", "-", "1", "1", "1"]]}
    df = parse_t86(payload, date(2026, 7, 24))
    assert len(df) == 1 and df["stock_id"].iloc[0] == "2330"
    assert df["inst_net"].iloc[0] == 2500.0
    assert df["trust_net"].iloc[0] == 500.0
    assert parse_t86({}, date(2026, 7, 24)).empty

def test_parse_margin_two_balance_columns():
    """MI_MARGN 解析：以兩個「今日餘額」欄依序對應融資/融券。"""
    from src.fetch.twse_chips import parse_margin
    payload = {"tables": [
        {"fields": ["日期", "說明"], "data": [["115/07/24", "x"]]},
        {"fields": ["股票代號", "股票名稱", "買進", "賣出", "今日餘額",
                    "買進", "賣出", "今日餘額"],
         "data": [["2330", "台積電", "1", "2", "5,000", "3", "4", "1,200"]]}]}
    df = parse_margin(payload, date(2026, 7, 24))
    assert len(df) == 1
    assert df["margin_bal"].iloc[0] == 5000.0
    assert df["short_bal"].iloc[0] == 1200.0
    assert parse_margin({}, date(2026, 7, 24)).empty

def test_pick_index_zero_not_falsy():
    """L58 回歸：代號欄位於索引 0 時不得被當成缺欄（`or` 串接的陷阱）。"""
    from src.fetch.twse_chips import parse_margin, parse_t86
    t86 = {"fields": ["證券代號", "三大法人買賣超股數"],
           "data": [["2317", "9,999"]]}
    assert parse_t86(t86, date(2026, 7, 24))["inst_net"].iloc[0] == 9999.0
    mg = {"fields": ["股票代號", "今日餘額", "今日餘額"],
          "data": [["2317", "100", "20"]]}
    got = parse_margin(mg, date(2026, 7, 24))
    assert got["margin_bal"].iloc[0] == 100.0 and got["short_bal"].iloc[0] == 20.0

def test_fetch_day_creates_cache_dir(tmp_path, monkeypatch):
    """L59 回歸：快取目錄不存在時必須自動建立（否則寫檔 FileNotFoundError）。

    此 bug 只在「解析成功、準備寫快取」時觸發——空資料路徑會提早 return，
    故單看解析測試無法發現。
    """
    import src.fetch.twse_chips as tc

    monkeypatch.chdir(tmp_path)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"fields": ["證券代號", "三大法人買賣超股數"],
                    "data": [["2330", "1,234"]]}

    class _Sess:
        def get(self, *a, **k):
            return _Resp()

    class _Cfg:
        request_timeout_sec = 5
        request_delay_sec = 0

    df = tc._fetch_day(tc._T86_URL, {}, tc.parse_t86, date(2026, 7, 23),
                       "T86", _Cfg(), _Sess())
    assert len(df) == 1 and df["inst_net"].iloc[0] == 1234.0
    cached = (tmp_path / "data" / "raw_chips"
              / f"T86_{tc.PARSER_VERSION}" / "20260723.csv")
    assert cached.exists(), "快取檔未寫出（目錄未建立）"
    # 二次呼叫走快取，且內容一致
    df2 = tc._fetch_day(tc._T86_URL, {}, tc.parse_t86, date(2026, 7, 23),
                        "T86", _Cfg(), _Sess())
    assert len(df2) == 1 and str(df2["stock_id"].iloc[0]) == "2330"

def test_t86_foreign_column_with_parenthetical(cfg=None):
    """L60 回歸：外資欄名為「外陸資買賣超股數(不含外資自營商)」——
    括號內含『自營』，不得被排除條件誤殺。"""
    from src.fetch.twse_chips import parse_t86
    payload = {"fields": ["證券代號", "證券名稱",
                          "外陸資買進股數(不含外資自營商)",
                          "外陸資賣出股數(不含外資自營商)",
                          "外陸資買賣超股數(不含外資自營商)",
                          "外資自營商買賣超股數",
                          "投信買賣超股數",
                          "自營商買賣超股數(自行買賣)",
                          "三大法人買賣超股數"],
               "data": [["2330", "台積電", "1", "2", "-8,888",
                         "11", "1,000", "22", "-5,000"]]}
    df = parse_t86(payload, date(2026, 7, 23))
    assert df["foreign_net"].iloc[0] == -8888.0            # 非 NaN、非自營欄
    assert df["trust_net"].iloc[0] == 1000.0
    assert df["inst_net"].iloc[0] == -5000.0

def test_describe_payload_surfaces_structure():
    """診斷輔助：能列出 top keys / fields / tables 概要（供 stat=OK 零列定位）。"""
    from src.fetch.twse_chips import describe_payload
    txt = describe_payload({"stat": "OK", "date": "20260723",
                            "tables": [{"title": "融資融券總量",
                                        "fields": ["項目", "買進"],
                                        "data": [["融資", "1"]]}]})
    assert "top_keys=" in txt and "n_tables=1" in txt and "融資融券總量" in txt
    assert "非 dict" in describe_payload(["x"])

def test_parse_margin_real_field_names_2026():
    """L61 回歸：融資融券彙總表實際欄名為「代號」（實抓確認 2026/7/24），
    兩個「今日餘額」依序為融資、融券。"""
    from src.fetch.twse_chips import parse_margin
    payload = {"stat": "OK", "date": "20260723", "tables": [
        {"title": "115年07月23日 信用交易統計",
         "fields": ["項目", "買進", "賣出", "現金(券)償還", "前日餘額",
                    "今日餘額"],
         "data": [["融資金額(仟元)", "1", "2", "3", "4", "5"]]},
        {"title": "115年07月23日 融資融券彙總 (全部)",
         "fields": ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額",
                    "今日餘額", "次一營業日限額", "買進", "賣出",
                    "現券償還", "前日餘額", "今日餘額", "次一營業日限額"],
         "data": [["2330", "台積電", "10", "20", "0", "5,100", "5,000",
                   "999", "1", "2", "0", "1,300", "1,200", "999"]]}]}
    df = parse_margin(payload, date(2026, 7, 23))
    assert len(df) == 1 and df["stock_id"].iloc[0] == "2330"
    assert df["margin_bal"].iloc[0] == 5000.0              # 融資今日餘額
    assert df["short_bal"].iloc[0] == 1200.0               # 融券今日餘額

def test_cache_path_is_parser_versioned():
    """L61：快取目錄綁解析器版本——解析邏輯改版後舊快取不得被沿用。"""
    import src.fetch.twse_chips as tc
    assert tc.PARSER_VERSION, "須定義 PARSER_VERSION"
    src_txt = __import__("pathlib").Path(
        "src/fetch/twse_chips.py").read_text(encoding="utf-8")
    assert "PARSER_VERSION}" in src_txt, "快取路徑未含解析器版本"
