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
    """籌碼資料 → 評分：法人與籌碼分項確實被計入（非缺值重分配）。"""
    monkeypatch.chdir(tmp_path)
    inst = pd.Series([2_000_000.0] * 6)                    # 連續買超
    vol = pd.Series([40_000_000.0] * 6)                    # 佔量 5%
    margin = pd.Series([10_000.0, 9_800.0, 9_600.0, 9_400.0, 9_200.0,
                        9_000.0])                          # 融資 -10%
    short = pd.Series([2_000.0] * 6)                       # 券資比 22%
    res = compute_simple_score(
        rev_yoy=0.25, pe=18.0, pe_hist=pd.Series(range(10, 130)),
        inst_net=inst, volume=vol, margin_bal=margin, short_bal=short,
        mv_short_dir=1, mv_mid_dir=1)
    assert res["n_available"] == 5, "五分項應全數可用"
    assert res["detail"]["institution"]["value"] is not None
    assert res["detail"]["margin"]["value"] is not None
    assert 0 <= res["score"] <= 100
    assert not res["capped_by_13mv"]
    # 同條件下 13MV 下彎必須壓到 20 以下（鐵律優先於高分）
    veto = compute_simple_score(
        rev_yoy=0.25, pe=18.0, pe_hist=pd.Series(range(10, 130)),
        inst_net=inst, volume=vol, margin_bal=margin, short_bal=short,
        mv_short_dir=1, mv_mid_dir=-1)
    assert veto["score"] <= 20.0 and veto["capped_by_13mv"]
    assert veto["score"] < res["score"]


def test_revenue_yoy_latest_two_files_only(tmp_path, monkeypatch):
    """輕量營收 YoY：僅抓 2 個月報檔、發布日對齊、月初自動退一月。"""
    monkeypatch.chdir(tmp_path)
    import src.fetch.mops_revenue as mr

    fetched: list[str] = []

    def fake_month(ym, cfg, session=None):
        fetched.append(ym)
        if ym in ("202607",):                              # 本月尚未發布
            return pd.DataFrame(columns=["stock_id", "revenue"])
        base = 100.0 if ym.startswith("2025") else 130.0   # 年增 30%
        return pd.DataFrame([{"stock_id": "2330", "revenue": base}])

    monkeypatch.setattr(mr, "fetch_revenue_month", fake_month)

    # 7/24（>=10 日）→ 可用 6 月；基期 2025-06
    yoy, note = mr.revenue_yoy_latest("2330", date(2026, 7, 24), object())
    assert abs(yoy - 0.30) < 1e-9, f"YoY 應為 30%，實得 {yoy}"
    assert "202606" in note and len(fetched) == 2, f"只該抓 2 檔，實抓 {fetched}"

    # 7/5（<10 日）→ 只能用 5 月（不得偷看 6 月）
    fetched.clear()
    yoy2, note2 = mr.revenue_yoy_latest("2330", date(2026, 7, 5), object())
    assert "202605" in note2 and abs(yoy2 - 0.30) < 1e-9

    # 標的不在月報中 → None 不猜
    fetched.clear()
    yoy3, note3 = mr.revenue_yoy_latest("9999", date(2026, 7, 24), object())
    assert yoy3 is None and "無營收列" in note3
