# -*- coding: utf-8 -*-
"""0050 量化分析面板（規格第三階段）。啟動：streamlit run app_streamlit.py

規格要求全數落實：下拉切換前十大｜Plotly OHLC + 波浪轉折（retrospective
灰色/realtime 藍色雙色區分）｜MV 潮汐副圖｜預測卡（看多空/信度/波浪警示/
固定風險聲明）｜VG 驗證狀態小卡（未通過如實顯示❌，不可省略）。
誠實原則：VG-6 未通過時預測卡強制掛紅色警語（模型無判別力，僅供演示）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from config.config import Config, DISCLAIMER
from config.phase2_config import Phase2Config
from config.phase3_config import Phase3Config, PHASE3_VERSION
from src.dashboard.charts import make_candles_figure, make_mv_figure
from src.dashboard.data_service import load_stock_view
from src.dashboard.model_service import load_model_pack, predict_latest
from src.dashboard.predictions_log import (prospective_progress,
                                           load_predictions_view,
                                           latest_prediction_per_stock)
from src.dashboard.export import to_csv_bytes, coverage_caption
from src.dashboard.stock_names import format_choice, load_names_from_holdings
from src.dashboard.vg_status import load_vg_status, vg6_blocking
from bootstrap_cloud import cloud_readiness, READINESS_GUIDE

st.set_page_config(page_title="0050 量化分析面板", layout="wide")
P1, P2, P3 = Config(), Phase2Config(), Phase3Config()

# 固定風險聲明（規格：必須固定顯示）
st.warning("⚠ " + P3.disclaimer_short + " " + DISCLAIMER)

st.title(f"0050 量化分析面板 v{PHASE3_VERSION}")

# ── 雲端就緒檢查（模型/快取缺失時明確引導，不空轉）──
_ready = cloud_readiness(P3.model_path, Path("holdings.csv"))
if not _ready["has_model"]:
    st.info(READINESS_GUIDE)

# ── 側欄：股票選擇（前十大 + 自選）──
holdings_path = Path("holdings.csv")
if holdings_path.exists():
    holding_ids = pd.read_csv(holdings_path, dtype=str)["stock_id"].tolist()
    names = load_names_from_holdings(holdings_path)   # 官方同源名稱，禁手寫
else:
    st.sidebar.error("找不到 holdings.csv，請置於同目錄")
    holding_ids, names = [], {}

# 自選股票（v3.9 修：輸入後自動加入+自動切換+清空輸入框——先前只加入
# 清單不切換，畫面停在原股票，使用者感受為「無動作」）
if "custom_ids" not in st.session_state:
    st.session_state.custom_ids = []
st.session_state.holding_ids = holding_ids


def _add_custom():
    code = st.session_state.custom_input.strip()
    st.session_state.custom_input = ""          # 清空輸入框
    if not code:
        return
    if not code.isdigit() or not (4 <= len(code) <= 6):
        st.session_state.custom_msg = f"「{code}」非有效代號（4–6 位數字）"
        return
    if (code not in st.session_state.custom_ids
            and code not in st.session_state.holding_ids):
        st.session_state.custom_ids.append(code)
    st.session_state.stock_select = code        # 自動切換到該股
    st.session_state.custom_msg = f"已切換至 {code}（首次載入需抓官方資料，請稍候）"


st.sidebar.markdown("**自選股票**（0050 以外亦可，技術圖照算）")
st.sidebar.text_input("輸入台股代號後按 Enter（例：2337）",
                      key="custom_input", max_chars=6, on_change=_add_custom)
_msg = st.session_state.pop("custom_msg", None)
if _msg:
    st.sidebar.info(_msg)
if st.session_state.custom_ids:
    drop = st.sidebar.multiselect("移除自選", st.session_state.custom_ids)
    if drop:
        st.session_state.custom_ids = [
            c for c in st.session_state.custom_ids if c not in drop]

ids = holding_ids + st.session_state.custom_ids
in_model_universe = set(holding_ids)     # 僅前十大在模型訓練範圍內


def _label(x):
    base = format_choice(x, names)
    return base if x in in_model_universe else f"{base}（自選）"


# 防護：選中的股票被移除時退回第一檔
if ids and st.session_state.get("stock_select") not in ids:
    st.session_state.stock_select = ids[0]
sid = st.sidebar.selectbox("選擇股票", ids, format_func=_label,
                           key="stock_select") if ids else None
st.sidebar.caption("資料源：TWSE 官方（逐月快取，禮貌延遲）；"
                   "首次載入新股票需抓取，較慢屬正常。自選股僅技術圖有效，"
                   "模型預測不適用（見下方說明）。")


@st.cache_data(ttl=3600, show_spinner="抓取官方日K與計算特徵中…")
def _view(stock_id: str, cache_key: str):
    return load_stock_view(stock_id, P1, P2, P3)


if sid:
    view = _view(sid, pd.Timestamp.today().strftime("%Y-%m-%d"))
    tail = view["ohlcv_tail"]
    lookback_start = pd.to_datetime(tail["date"].iloc[0])

    # ── 主圖 + 副圖 ──
    is_custom = sid not in in_model_universe
    if is_custom:
        st.warning(f"⚠ {format_choice(sid, names)} 為自選股，不在 0050 前十大"
                   "模型訓練範圍內：以下 K線／波浪／MV 潮汐等技術圖照官方資料"
                   "計算有效；但模型預測卡不適用此股（模型未見過它），將隱藏。")
    st.subheader(f"{_label(sid)}｜末根K {view['last_bar_date']}"
                 f"｜收盤 {view['last_close']}")
    st.caption(coverage_caption(view["ohlcv"]))
    dl1, dl2, _sp = st.columns([1, 1, 2])
    dl1.download_button(
        "⬇ 下載K線 CSV", to_csv_bytes(view["ohlcv"]),
        file_name=f"{sid}_ohlcv.csv", mime="text/csv",
        help="十年官方日K（開高低收量），Excel 可直接開啟")
    _feat_export = view["features"].drop(
        columns=[c for c in ("fwd_return_gross", "fwd_return_net", "label_up")
                 if c in view["features"].columns])
    dl2.download_button(
        "⬇ 下載特徵 CSV", to_csv_bytes(_feat_export),
        file_name=f"{sid}_features.csv", mime="text/csv",
        help="技術特徵全表（MV潮汐/RSI/MACD/波浪標籤等，不含未來報酬欄）")
    st.plotly_chart(make_candles_figure(tail, view["pivots_retro"],
                                        view["pivots_rt"], lookback_start),
                    width='stretch')
    st.caption("轉折點：灰=retrospective（回溯視角，僅供視覺參考，禁入模型）"
               "／藍=realtime（僅用已確認轉折，可用於決策層）")
    st.plotly_chart(make_mv_figure(tail, view["mv"], lookback_start, P1),
                    width='stretch')

    # ── 預測卡（自選股不適用；VG-6 蓋過警語優先）──
    st.subheader("模型預測卡")
    if is_custom:
        st.info("此為自選股，不在模型訓練範圍（模型僅以 0050 前十大訓練），"
                "故不提供模型預測。上方技術圖（K線/波浪/MV 潮汐）仍有效，"
                "可據方法論人工研判。")
        model = None
    else:
        blocked, msg = vg6_blocking(P3.report_json)
        if blocked:
            st.error("🛑 " + msg)
        model, bundle = load_model_pack(P3.model_path)
        if model is None:
            st.info("模型包不存在——先執行 run_phase2.py 產生。")
    if (not is_custom) and model is not None:
        try:
            pred = predict_latest(model, bundle, view["features"])
        except Exception as exc:                       # 特徵欄不符等，明確呈現
            st.error(f"模型套用失敗（不靜默）：{exc}")
            pred = None
        if pred:
            st.markdown(f"**本股即時值**（隨 {format_choice(sid, names)} 變動）")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("方向", "看多" if pred["pick"] else "看空/觀望")
            c2.metric("P(未來{}日淨報酬>0)".format(P2.forward_return_days),
                      f"{pred['proba_up']:.1%}")
            feats_last = view["features"].tail(1).iloc[0]
            c3.metric("realtime 波浪", str(feats_last["wave_label_realtime"]))
            veto = bool(feats_last.get("mv_mid_veto_active", False))
            c4.metric("13MV 核心否決線", "⚠ 下彎(否決)" if veto else "未觸發")
            if veto:
                st.error("⚠ 波浪警示：13MV 下彎＝方法論絕對否決訊號，"
                         "凌駕任何模型/波浪判讀。")
            st.caption(f"模型：{pred['model_tag']}｜特徵基準日 {pred['as_of']}"
                       f"｜{P3.disclaimer_short}")

    # ── VG 驗證狀態小卡（全模型層級，對每檔股票相同）──
    st.subheader("驗證關卡狀態（全模型層級，非單股）")
    st.caption("這六關檢驗的是「整個模型與規則訊號的統計可信度」，"
               "十檔以同一模型訓練，故每檔股票顯示相同——這是設計、非錯誤。"
               "單股即時判讀請看上方預測卡（方向/機率/波浪/13MV 逐股變動）。"
               "注意：此驗證僅涵蓋 0050 前十大模型；自選股不適用模型，"
               "本區狀態與自選股無關。")
    cards = load_vg_status(P3.report_json)
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        icon = "✅" if c["passed"] else ("❌" if c["passed"] is False else "⚠")
        col.metric(c["gate"], icon)
        col.caption(c["note"])
    # 科學結論一句話摘要（讓 ❌ 有脈絡，不被誤讀為「系統壞了」）
    st.info("科學結論（Phase 2 定案）：VG-3❌＝規則訊號無可證明的優勢；"
            "VG-6❌＝模型 AUC≈0.5 無判別力。兩者為「誠實的否定結論」，"
            "系統正確攔下了假訊號，非故障。詳見 PHASE2_CONCLUSION.md。")

    # ── 前瞻協定進度（Model v2 最終閘）──
    st.subheader("前瞻驗證進度（Model v2 最終閘）")
    prog = prospective_progress(P3.predictions_csv, P2.forward_return_days)
    st.progress(min(1.0, prog["n_independent"] / P3.min_prospective_samples),
                text=f"獨立樣本 {prog['n_independent']} / "
                     f"{P3.min_prospective_samples}（紀錄 {prog['n_rows']} 列）")
    st.caption("每日收盤後由 GitHub Actions 自動執行 daily_update.py 累積；"
               "達標前不得對 Model v2 下結論（預先宣告規則）。")

    # ── 每日前瞻紀錄（自動讀取 Actions 累積的 predictions.csv）──
    st.subheader("每日前瞻紀錄（自動更新）")
    latest = latest_prediction_per_stock(P3.predictions_csv)
    if len(latest) == 0:
        st.info("尚無前瞻紀錄。GitHub Actions「每日前瞻更新」每交易日 22:00 "
                "自動累積；也可在 repo 的 Actions 分頁手動 Run workflow 立即產生。")
    else:
        st.markdown("**各股最新預測**（每日自動更新，非即時報價）")
        show = latest[["stock_id", "last_bar_date", "close",
                       "proba_up", "pick"]].copy()
        show["last_bar_date"] = show["last_bar_date"].dt.strftime("%Y-%m-%d")
        show["proba_up"] = (show["proba_up"] * 100).round(1).astype(str) + "%"
        show["pick"] = show["pick"].map({True: "看多", False: "觀望"})
        show.columns = ["代碼", "資料日", "收盤", "P(漲)", "方向"]
        st.dataframe(show, hide_index=True)
        # 選定股票的預測歷史趨勢
        hist = load_predictions_view(P3.predictions_csv, sid)
        if len(hist) >= 2:
            import plotly.graph_objects as _go
            fig = _go.Figure(_go.Scatter(
                x=hist["last_bar_date"], y=hist["proba_up"],
                mode="lines+markers", name="P(漲)",
                line=dict(color="#1f77b4")))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=28, b=10),
                              yaxis_title="P(未來5日淨報酬>0)",
                              title=f"{_label(sid)} 前瞻預測歷史")
            st.plotly_chart(fig, width='stretch')
            st.caption("此圖為模型每日輸出的機率軌跡；VG-6 現況下模型無判別力，"
                       "僅供前瞻管線演示，達 30 獨立樣本後才做最終裁決。")
        elif sid:
            st.caption(f"{format_choice(sid, names)} 目前僅 {len(hist)} 筆紀錄，"
                       "累積 2 筆以上才顯示趨勢圖。")
