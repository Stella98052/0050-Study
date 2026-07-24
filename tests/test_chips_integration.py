# -*- coding: utf-8 -*-
"""籌碼面端到端整合測試（v3.26）。

所有 fixture 均取自**使用者本機實抓回傳的真實結構**（2026-07-24），
非文件推測：
- T86 欄名含「外陸資買賣超股數(不含外資自營商)」等括號註記
- MI_MARGN 為 tables[2]，個股表欄名為「代號」，兩個「今日餘額」
  依序為融資/融券，實測 1287 檔
- 真實數值：2330 於 2026-07-23 三大法人 -5,519,232、投信 -943,258
"""
from __future__ import annotations

from datetime import date

import pandas as pd

import src.fetch.twse_chips as tc
from src.score.simple_score import compute_simple_score

# ── 真實回應結構（實抓節錄）──
T86_FIELDS = ["證券代號", "證券名稱",
              "外陸資買進股數(不含外資自營商)",
              "外陸資賣出股數(不含外資自營商)",
              "外陸資買賣超股數(不含外資自營商)",
              "外資自營商買進股數", "外資自營商賣出股數",
              "外資自營商買賣超股數",
              "投信買進股數", "投信賣出股數", "投信買賣超股數",
              "自營商買賣超股數",
              "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)",
              "自營商買賣超股數(自行買賣)",
              "自營商買進股數(避險)", "自營商賣出股數(避險)",
              "自營商買賣超股數(避險)",
              "三大法人買賣超股數"]

MARGIN_FIELDS = ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額",
                 "今日餘額", "次一營業日限額", "買進", "賣出", "現券償還",
                 "前日餘額", "今日餘額", "次一營業日限額"]


def _t86_payload(inst: str, trust: str, foreign: str) -> dict:
    row = ["2330", "台積電", "1", "2", foreign, "3", "4", "5",
           "6", "7", trust, "8", "9", "10", "11", "12", "13", "14", inst]
    assert len(row) == len(T86_FIELDS)
    return {"stat": "OK", "fields": T86_FIELDS,
            "data": [row, ["合計", "-"] + ["0"] * 17]}


def _margin_payload(d: str, margin: str, short: str) -> dict:
    return {"stat": "OK", "date": d, "tables": [
        {"title": f"{d} 信用交易統計",
         "fields": ["項目", "買進", "賣出", "現金(券)償還", "前 日餘額",
                    "今日餘額"],
         "data": [["融資金額(仟元)", "1", "2", "3", "4", "5"]]},
        {"title": f"{d} 融資融券彙總 (全部)",
         "fields": MARGIN_FIELDS,
         "data": [["2330", "台積電", "1", "2", "0", "9,999", margin, "0",
                   "3", "4", "0", "888", short, "0"],
                  ["2317", "鴻海", "1", "2", "0", "1", "500", "0",
                   "3", "4", "0", "1", "50", "0"]]}]}


def test_parse_t86_real_fields_all_three_columns():
    """真實欄名下三個法人欄位皆須正確（外資曾因括號含『自營』被誤殺）。"""
    df = tc.parse_t86(_t86_payload("-5,519,232", "-943,258", "1,234,567"),
                      date(2026, 7, 23))
    assert len(df) == 1                                    # 「合計」列剔除
    r = df.iloc[0]
    assert r["stock_id"] == "2330"
    assert r["inst_net"] == -5519232.0                     # 三大法人（實測值）
    assert r["trust_net"] == -943258.0                     # 投信（實測值）
    assert r["foreign_net"] == 1234567.0                   # 外陸資，非自營欄


def test_parse_margin_real_structure_multi_rows():
    """真實 tables[2] 結構：跳過統計表、由個股表取兩個『今日餘額』。"""
    df = tc.parse_margin(_margin_payload("115年07月23日", "5,000", "1,200"),
                         date(2026, 7, 23))
    assert len(df) == 2
    tsmc = df[df["stock_id"] == "2330"].iloc[0]
    assert tsmc["margin_bal"] == 5000.0 and tsmc["short_bal"] == 1200.0


def test_fetch_chips_recent_end_to_end(tmp_path, monkeypatch):
    """端到端：多日抓取→快取→合併→欄位齊備（含休市日略過）。"""
    monkeypatch.chdir(tmp_path)
    trading = {date(2026, 7, 21), date(2026, 7, 22), date(2026, 7, 23)}

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._p

    class _Sess:
        calls = 0

        def get(self, url, params=None, headers=None, timeout=None):
            _Sess.calls += 1
            ymd = params["date"]
            d = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
            if d not in trading:                           # 休市/未發布
                return _Resp({"stat": "很抱歉，沒有符合條件的資料!"})
            if "T86" in url:
                return _Resp(_t86_payload("1,000,000", "50,000", "800,000"))
            return _Resp(_margin_payload(f"{d}", "5,000", "1,500"))

    class _Cfg:
        request_timeout_sec = 5
        request_delay_sec = 0

    out = tc.fetch_chips_recent(["2330"], date(2026, 7, 24), 6, _Cfg(),
                                session=_Sess())
    assert len(out) == 3, f"應得 3 個交易日，實得 {len(out)}"
    for col in ("inst_net", "trust_net", "foreign_net",
                "margin_bal", "short_bal"):
        assert col in out.columns, f"缺欄 {col}"
        assert out[col].notna().all(), f"{col} 有 NaN"
    assert (out["stock_id"] == "2330").all()               # 其他股已濾除
    assert out["date"].is_monotonic_increasing

    # 快取生效：二次呼叫不再發 HTTP（僅休市日重試）
    before = _Sess.calls
    out2 = tc.fetch_chips_recent(["2330"], date(2026, 7, 24), 6, _Cfg(),
                                 session=_Sess())
    assert len(out2) == 3
    assert _Sess.calls - before < 6, "快取未生效"
    # 快取路徑含解析器版本（L61）
    assert (tmp_path / "data" / "raw_chips"
            / f"T86_{tc.PARSER_VERSION}").exists()


def test_chips_to_score_pipeline(tmp_path, monkeypatch):
    """籌碼資料 → 六因子評分：法人與籌碼分項確實計入（非缺值重分配）。"""
    monkeypatch.chdir(tmp_path)
    inst = pd.Series([2_000_000.0] * 6)
    vol = pd.Series([40_000_000.0] * 6)
    margin = pd.Series([10_000.0, 9_800.0, 9_600.0, 9_400.0, 9_200.0, 9_000.0])
    short = pd.Series([2_000.0] * 6)
    kw = dict(rev_yoy=0.25, eps_accel=0.10,
              margins={"gross_yoy": 0.02, "op_yoy": 0.015}, roe=0.18,
              fcf=1e9, inst_net=inst, volume=vol, margin_bal=margin,
              short_bal=short, pe=18.0, pe_hist=pd.Series(range(10, 130)))
    res = compute_simple_score(**kw, mv_short_dir=1, mv_mid_dir=1)
    assert res["n_available"] == 6
    assert res["detail"]["institution"]["value"] is not None
    assert res["detail"]["margin"]["value"] is not None
    assert res["detail"]["quality"]["value"] is not None
    assert 0 <= res["score"] <= 100 and not res["capped_by_13mv"]
    veto = compute_simple_score(**kw, mv_short_dir=1, mv_mid_dir=-1)
    assert veto["score"] <= 20.0 and veto["score"] < res["score"]


def test_financial_factors_from_quarters():
    """財務因子（輸入為累計數，符合 MOPS 實況）：
    EPS 單季 YoY 與加速度、三率 YoY、ROE(TTM)、FCF=ΣOCF−Σcapex。"""
    from src.features.fin_factors import (eps_growth_and_accel,
                                          free_cash_flow, roe_ttm,
                                          three_margins)
    # 欄位皆為「年初至該季累計」（equity 除外，為存量）
    cols = ["year", "season", "eps", "revenue", "gross_profit", "op_income",
            "net_income", "equity", "ocf", "capex"]
    rows = [
        (2024, 3, 18, 2000, 900, 700, 600, 4000, 700, 250),
        (2024, 4, 26, 2800, 1300, 1000, 850, 4200, 1000, 380),
        (2025, 1, 8, 800, 400, 300, 200, 4500, 300, 100),
        (2025, 2, 17, 1700, 850, 640, 420, 4600, 620, 220),
        (2025, 3, 27, 2700, 1350, 1000, 660, 4800, 960, 350),
        (2025, 4, 38, 3800, 1900, 1400, 950, 4900, 1300, 480),
        (2026, 1, 12, 1000, 600, 450, 300, 5000, 500, 200),
    ]
    fin = pd.DataFrame(rows, columns=cols).astype(float)

    # 單季 EPS：2026Q1=12 vs 2025Q1=8 → +50%
    g, accel = eps_growth_and_accel(fin)
    assert abs(g - 0.50) < 1e-9
    # 上一季 2025Q4 單季=38−27=11 vs 2024Q4 單季=26−18=8 → +37.5%
    assert abs(accel - (0.50 - 0.375)) < 1e-9

    m = three_margins(fin)
    assert abs(m["gross"] - 0.60) < 1e-9                   # 600/1000（單季）
    assert abs(m["gross_yoy"] - 0.10) < 1e-9               # 60% vs 50%
    assert abs(m["net"] - 0.30) < 1e-9

    # ROE：近四季單季淨利 300+290+240+220 = 1050 ÷ 最新權益 5000
    assert abs(roe_ttm(fin) - 1050 / 5000) < 1e-9

    # FCF：ΣOCF(500+340+340+320) − Σcapex(200+130+130+120)
    assert abs(free_cash_flow(fin) - (1500 - 580)) < 1e-9

    empty = pd.DataFrame(columns=["year", "season"])
    assert eps_growth_and_accel(empty) == (None, None)
    assert roe_ttm(empty) is None and free_cash_flow(empty) is None


def test_publication_date_alignment_no_lookahead():
    """公告日對齊：季底當天不得可用，須等法定期限（防前視核心）。"""
    from src.fetch.mops_financials import available_quarters, publication_date
    assert publication_date(2026, 1) == date(2026, 5, 15)
    assert publication_date(2025, 4) == date(2026, 3, 31)   # 年報次年
    # 5/14 尚不可用 Q1；5/15 起可用
    assert (2026, 1) not in available_quarters(date(2026, 5, 14))
    assert (2026, 1) in available_quarters(date(2026, 5, 15))
    # 季底當天（3/31）絕不可用該季（Q1 尚未結束更不可能）
    assert (2026, 1) not in available_quarters(date(2026, 3, 31))
    q = available_quarters(date(2026, 7, 24), 4)
    assert q == sorted(q, key=lambda x: (x[0], x[1]), reverse=True)


def test_cumulative_to_single_quarter_real_2330():
    """L64 鎖定：MOPS 季報為累計數，須轉單季——以 2330 實抓數字驗證。

    實抓（2026/7/24）2025 Q2/Q3/Q4 營收 1.77/2.76/3.81 兆逐季遞增＝累計。
    若直接相加四季計 ROE，會膨脹逾一倍（71% vs 實際約 30%）。
    """
    from src.features.fin_factors import (roe_ttm, three_margins,
                                          to_single_quarter)
    fin = pd.DataFrame([
        {"year": 2026, "season": 1, "eps": 22.08, "revenue": 1.134103e9,
         "gross_profit": 7.512954e8, "op_income": 6.589661e8,
         "net_income": 5.728013e8, "equity": 5.932389e9, "ocf": 6.989763e8},
        {"year": 2025, "season": 4, "eps": 66.26, "revenue": 3.809054e9,
         "gross_profit": 2.281294e9, "op_income": 1.936092e9,
         "net_income": 1.715397e9, "equity": 5.460795e9, "ocf": 2.274976e9},
        {"year": 2025, "season": 3, "eps": 46.75, "revenue": 2.762964e9,
         "gross_profit": 1.629307e9, "op_income": 1.371189e9,
         "net_income": 1.209981e9, "equity": 5.035578e9, "ocf": 1.549467e9},
        {"year": 2025, "season": 2, "eps": 29.31, "revenue": 1.773046e9,
         "gross_profit": 1.040764e9, "op_income": 8.705044e8,
         "net_income": 7.582261e8, "equity": 4.616632e9, "ocf": 1.122638e9},
    ])
    sq = to_single_quarter(fin)
    q4 = sq[(sq["year"] == 2025) & (sq["season"] == 4)].iloc[0]
    assert abs(q4["revenue"] - (3.809054e9 - 2.762964e9)) < 1.0
    q1 = sq[(sq["year"] == 2026) & (sq["season"] == 1)].iloc[0]
    assert abs(q1["revenue"] - 1.134103e9) < 1.0           # Q1 累計即單季
    # 缺同年上季 → NaN 不猜（此處缺 2025Q1）
    q2 = sq[(sq["year"] == 2025) & (sq["season"] == 2)].iloc[0]
    assert pd.isna(q2["revenue"])
    # equity 為存量，不得被差分
    assert abs(q4["equity"] - 5.460795e9) < 1.0

    m = three_margins(fin)
    assert 0.60 < m["gross"] < 0.72, f"毛利率 {m['gross']:.1%} 偏離實況"
    roe = roe_ttm(fin)
    assert 0.20 < roe < 0.50, f"ROE {roe:.1%} 應在合理區間（未修正時 >0.7）"
