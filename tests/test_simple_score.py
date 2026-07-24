# -*- coding: utf-8 -*-
"""簡單評分模型與籌碼資料鎖定測試（v3.25）。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.score.simple_score import (WEIGHTS, compute_simple_score,
                                    score_institution, score_margin,
                                    score_revenue, score_tide,
                                    score_valuation)


def test_score_components_bounds_and_missing():
    """各分項：值域 0–1、缺資料回 None（觸發權重重分配，不填預設值）。"""
    assert score_revenue(0.40)[0] == 1.0
    assert score_revenue(0.0)[0] == 0.0
    assert score_revenue(2.0)[0] == 1.0                    # 上限截斷
    assert score_revenue(None)[0] is None
    assert score_valuation(None, None)[0] is None
    assert score_valuation(15.0, pd.Series([10.0] * 30))[0] is None  # 樣本不足
    hist = pd.Series(range(10, 110))                       # 100 個歷史值
    v_cheap, _ = score_valuation(15.0, hist)
    v_rich, _ = score_valuation(105.0, hist)
    assert v_cheap > v_rich                                # 便宜得分較高


def test_score_institution_streak_and_ratio():
    """法人動能：連買天數＋佔量比；資料不足回 None。"""
    net = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])   # 連買5日
    vol = pd.Series([10000.0] * 5)
    v, txt = score_institution(net, vol)
    assert 0.0 <= v <= 1.0 and "連續買超 5 日" in txt
    sell = pd.Series([-100.0] * 5)
    v2, _ = score_institution(sell, vol)
    assert v2 < v                                          # 連賣分數較低
    assert score_institution(pd.Series([1.0]), vol)[0] is None


def test_score_margin_structure():
    """籌碼結構：融資下降得分高於融資上升。"""
    down = pd.Series([1000.0, 950.0, 900.0, 880.0, 850.0])
    up = pd.Series([850.0, 880.0, 900.0, 950.0, 1000.0])
    short = pd.Series([200.0])
    v_down, _ = score_margin(down, short)
    v_up, _ = score_margin(up, short)
    assert v_down > v_up
    assert score_margin(None, short)[0] is None


def test_score_tide_follows_13mv_rules():
    """量價動能嚴格對應 13MV 三結論。"""
    assert score_tide(1, 1, False)[0] == 1.0
    assert score_tide(1, 0, False)[0] == 0.5
    assert score_tide(0, 0, False)[0] == 0.25
    assert score_tide(1, 1, True)[0] == 0.0                # 否決凌駕


def test_compute_score_weight_redistribution_and_veto_cap():
    """總分：缺項只用可用權重重分配；13MV 下彎硬性封頂 20。"""
    full = compute_simple_score(
        rev_yoy=0.40, pe=10.0, pe_hist=pd.Series(range(10, 110)),
        inst_net=pd.Series([100.0] * 5), volume=pd.Series([1000.0] * 5),
        margin_bal=pd.Series([1000.0, 900.0, 880.0, 860.0, 850.0]),
        short_bal=pd.Series([300.0]), mv_short_dir=1, mv_mid_dir=1)
    assert full["n_available"] == 5 and 0 <= full["score"] <= 100
    assert "非經統計驗證" in full["disclaimer"]

    partial = compute_simple_score(rev_yoy=0.40, mv_short_dir=1, mv_mid_dir=1)
    assert partial["n_available"] == 2                     # 僅兩項可用
    assert set(partial["weight_used"]) == {"revenue", "tide"}
    assert partial["score"] == 100.0                       # 兩項皆滿分

    veto = compute_simple_score(
        rev_yoy=0.40, inst_net=pd.Series([500.0] * 5),
        volume=pd.Series([1000.0] * 5), mv_short_dir=1, mv_mid_dir=-1)
    assert veto["capped_by_13mv"] and veto["score"] <= 20.0

    assert compute_simple_score()["n_available"] >= 1       # tide 恆可用


def test_every_component_reports_calc_logic():
    """治理：每一分項都必須附帶可解釋的計算邏輯字串。"""
    r = compute_simple_score(rev_yoy=0.2, mv_short_dir=1, mv_mid_dir=1)
    for k, d in r["detail"].items():
        assert isinstance(d["calc_logic"], str) and len(d["calc_logic"]) > 5
    assert sum(WEIGHTS.values()) == 100.0


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
