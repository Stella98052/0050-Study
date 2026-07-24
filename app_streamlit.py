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
                                           latest_prediction_per_stock,
                                           staleness_days)
from src.dashboard.export import to_csv_bytes, coverage_caption
from src.dashboard.stock_names import (format_choice,
                                       fetch_all_listed_names,
                                       load_names_from_holdings)
from src.custom.fetch_and_save import load_watchlist
from src.dashboard.tidal import DIR_TEXT, TIDAL_DISCLAIMER, tidal_state
from src.dashboard.interpret import interpret_card
from src.dashboard.watchlist import parse_watch_param, serialize_watch
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


@st.cache_data(ttl=86400, show_spinner=False)
def _all_listed_names():
    """官方全市場代號→簡稱（t187ap03_L；自選股名稱用，快取一天）。"""
    return fetch_all_listed_names()


if holdings_path.exists():
    holding_ids = pd.read_csv(holdings_path, dtype=str)["stock_id"].tolist()
    # v3.20：官方全市場為底、holdings 同源名稱覆蓋（皆官方，禁手寫）
    names = {**_all_listed_names(), **load_names_from_holdings(holdings_path)}
else:
    st.sidebar.error("找不到 holdings.csv，請置於同目錄")
    holding_ids, names = [], {}

# 自選股票（v3.9 修：輸入後自動加入+自動切換+清空輸入框——先前只加入
# 清單不切換，畫面停在原股票，使用者感受為「無動作」）
if "custom_ids" not in st.session_state:
    # v3.20：repo 凍結清單（永久）∪ 網址參數（書籤）——雲端檔案系統
    # 每次重新部署即清空，唯一跨部署的永久儲存＝repo 內 custom_watchlist.csv
    _repo_wl = load_watchlist(Path("custom_watchlist.csv"))
    _url_wl = parse_watch_param(st.query_params.get("watch", ""))
    st.session_state.custom_ids = list(dict.fromkeys(_repo_wl + _url_wl))
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

# v3.11：清單同步至網址參數（idempotent；書籤即存檔）
if st.session_state.custom_ids:
    st.query_params["watch"] = serialize_watch(st.session_state.custom_ids)
elif "watch" in st.query_params:
    del st.query_params["watch"]
if st.session_state.custom_ids:
    st.sidebar.caption("💾 保存清單兩種方式：①將目前網址加入書籤（本瀏覽器）"
                       "②把代號寫入 repo 的 custom_watchlist.csv 後 push"
                       "（永久，所有裝置開啟即載入）。")
    # ── 自選股資料儲存（面板內，v3.17）──
    with st.sidebar.expander("📦 儲存自選股資料"):
        _val = st.checkbox("含官方估值（本益比/殖利率/淨值比）",
                           value=False, key="wl_val")
        if st.button("產生下載包", key="wl_build"):
            from src.custom.fetch_and_save import build_watchlist_zip
            with st.spinner("以官方管線產出中（首次含抓取，請稍候）…"):
                zb, sums = build_watchlist_zip(
                    st.session_state.custom_ids, P1, P2, with_valuation=_val)
            st.session_state.wl_zip = zb
            st.session_state.wl_sums = sums
        if st.session_state.get("wl_zip"):
            for r in st.session_state.get("wl_sums", []):
                st.caption(("✓" if r["ok"] else "✗")
                           + f" {r['stock_id']}：{r['msg']}")
            st.download_button(
                "⬇ 下載自選股資料包 (ZIP)", st.session_state.wl_zip,
                file_name="watchlist_data.zip", mime="application/zip",
                key="wl_dl",
                help="每檔含 K線/特徵 CSV（無未來報酬欄），與模型池隔離")

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


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_score_for(sid: str, _fl_dict: dict):
    """單股簡易評分（籌碼近 20 日、估值近三年；快取一小時）。"""
    from datetime import date, timedelta
    from src.fetch.twse_bwibbu import fetch_valuation_history
    from src.fetch.twse_chips import fetch_chips_recent
    from src.score.simple_score import compute_simple_score
    end = date.today()
    try:
        chips = fetch_chips_recent([sid], end, 30, P1)
        chips = chips[chips["stock_id"] == sid].sort_values("date")
    except Exception:                                      # noqa: BLE001
        chips = pd.DataFrame()
    try:
        val = fetch_valuation_history(sid, end - timedelta(days=1095), end, P1)
    except Exception:                                      # noqa: BLE001
        val = pd.DataFrame()
    view = _view(sid, pd.Timestamp.today().strftime("%Y-%m-%d"))
    vol = view["ohlcv"].sort_values("date")["volume"].tail(30)
    # 營收動能：特徵矩陣不含 rev_yoy（僅稽核腳本才加），故此處實抓 MOPS
    # 官方月報並套用發布日對齊（M 月營收次月 10 日起可用）
    try:
        from src.fetch.mops_revenue import revenue_yoy_latest
        rev, _rev_note = revenue_yoy_latest(sid, end, P1)
    except Exception:                                      # noqa: BLE001
        rev = None
    return compute_simple_score(
        rev_yoy=rev,
        pe=(float(val["pe_ratio"].dropna().iloc[-1])
            if len(val) and val["pe_ratio"].notna().any() else None),
        pe_hist=(val["pe_ratio"] if len(val) else None),
        inst_net=(chips["inst_net"] if "inst_net" in chips else None),
        volume=vol,
        margin_bal=(chips["margin_bal"] if "margin_bal" in chips else None),
        short_bal=(chips["short_bal"] if "short_bal" in chips else None),
        mv_short_dir=int(_fl_dict.get("mv_short_direction", 0)),
        mv_mid_dir=int(_fl_dict.get("mv_mid_direction", 0)),
        veto=bool(_fl_dict.get("mv_mid_veto_active", False)))


def _render_tidal_snapshot(fl, _itp_proba=None, _itp_custom=False,
                           _itp_sid="x"):
    """潮汐快照+解讀結論（前十大與自選股共用；v3.14 P1、v3.19 解讀）。"""
    sd = int(fl.get("mv_short_direction", 0))
    md = int(fl.get("mv_mid_direction", 0))
    veto = bool(fl.get("mv_mid_veto_active", False))
    stt = tidal_state(sd, md, veto)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("5MV 方向", DIR_TEXT.get(sd, "—"))
    c2.metric("13MV 方向", DIR_TEXT.get(md, "—"))
    c3.metric("潮汐狀態", f"{stt['emoji']} {stt['label']}",
              help=stt["desc"])
    _b = fl.get("mv_bias")
    burst = bool(fl.get("is_volume_burst", False))
    c4.metric("量能乖離率", (f"{_b:.1%}" + ("🔥爆發" if burst else ""))
              if pd.notna(_b) else "—")
    if stt["emoji"] == "🔴":
        st.error("⚠ " + stt["desc"] + "（凌駕任何模型/波浪判讀）")
    st.caption(TIDAL_DISCLAIMER)
    # ── 解讀結論（v3.19：規則驅動，零猜想）──
    # ── 簡易評分（v3.25：基本面＋籌碼面透明加權，按鈕觸發）──
    with st.expander("🧮 簡易評分（基本面＋籌碼面）"):
        st.caption("五分項透明加權：營收動能25／估值位階20／法人動能30／"
                   "籌碼結構15／量價動能10。缺項權重重分配，不填預設值。"
                   "**決策輔助檢核分數，非經統計驗證之預測器。**")
        if st.button("計算評分（抓取近 20 日籌碼，約 15–40 秒）",
                     key=f"score_btn_{_itp_sid}"):
            with st.spinner("抓取官方籌碼與估值資料中…"):
                st.session_state[f"score_{_itp_sid}"] = _compute_score_for(
                    _itp_sid, {k: fl.get(k) for k in
                               ("rev_yoy", "mv_short_direction",
                                "mv_mid_direction", "mv_mid_veto_active")})
        _res = st.session_state.get(f"score_{_itp_sid}")
        if _res:
            if _res.get("score") is None:
                st.warning("無足夠官方資料可評分。")
            else:
                st.metric("綜合評分", f"{_res['score']} / 100",
                          help=f"採用 {_res['n_available']}/5 分項")
                if _res.get("capped_by_13mv"):
                    st.error("⚠ 13MV 下彎：方法論鐵律，總分強制封頂 20")
                for _k, _lab in (("revenue", "營收動能"),
                                 ("valuation", "估值位階"),
                                 ("institution", "法人動能"),
                                 ("margin", "籌碼結構"),
                                 ("tide", "量價動能")):
                    _d = _res["detail"][_k]
                    _v = _d["value"]
                    st.markdown(
                        f"- **{_lab}**："
                        + (f"{_v:.2f}" if _v is not None else "—")
                        + f"　<span style='color:gray;font-size:0.85em'>"
                          f"{_d['calc_logic']}</span>",
                        unsafe_allow_html=True)
                st.caption(_res["disclaimer"])

    itp = interpret_card(
        proba=_itp_proba, mv_short_dir=sd, mv_mid_dir=md, veto=veto,
        bias=(float(_b) if pd.notna(_b) else None), burst=burst,
        wave_label=str(fl.get("wave_label_realtime", "")),
        is_custom=_itp_custom)
    with st.expander(f"📋 解讀結論：{itp['emoji']} {itp['state']}",
                     expanded=True):
        st.markdown(f"**方法論判讀**：{itp['method']}")
        st.markdown(f"**模型數字定位**：{itp['model_note']}")
        st.markdown(f"**行動語意**：{itp['action']}")
        st.caption(itp["disclaimer"])


if sid:
    try:
        view = _view(sid, pd.Timestamp.today().strftime("%Y-%m-%d"))
    except Exception as exc:                     # v3.13：不噴 traceback
        st.error(f"⚠ 無法載入 {sid}：{exc}")
        st.info("提示：上櫃/興櫃股票與錯誤代號目前不支援；"
                "可從「移除自選」把此代號刪除。")
        st.stop()
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
                "故不提供模型預測；以下技術快照為官方資料即時計算，有效。")
        fl = view["features"].tail(1).iloc[0]
        st.metric("realtime 波浪", str(fl["wave_label_realtime"]))
        _render_tidal_snapshot(fl, _itp_custom=True, _itp_sid=sid)
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
            st.markdown("**潮汐快照**")
            _render_tidal_snapshot(feats_last, _itp_proba=pred["proba_up"],
                                   _itp_sid=sid)

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
        _stale = staleness_days(P3.predictions_csv)
        if _stale is not None and _stale > 4:
            st.warning(f"⏰ 前瞻紀錄已 {_stale} 天未更新——watchdog 每晚 21:10 "
                       f"自動檢查並補跑；亦可至 GitHub Actions 手動 Run "
                       f"「每日前瞻更新」。連假期間屬正常。")
        st.markdown("**各股最新預測**（每日自動更新，非即時報價）")
        show = latest[["stock_id", "last_bar_date", "close",
                       "proba_up", "pick"]].copy()
        show["stock_id"] = show["stock_id"].map(
            lambda x: format_choice(x, names))          # v3.12：代碼帶名稱
        show["last_bar_date"] = show["last_bar_date"].dt.strftime("%Y-%m-%d")
        show["proba_up"] = (show["proba_up"] * 100).round(1).astype(str) + "%"
        show["pick"] = show["pick"].map({True: "看多", False: "觀望"})
        show.columns = ["代碼", "資料日", "收盤", "P(漲)", "方向"]
        st.dataframe(show, hide_index=True)
        # 選定股票的預測歷史趨勢
        _sid_custom = sid not in in_model_universe
        if _sid_custom:
            # 自選股：顯示每日技術快照歷史（模型紀錄不適用，v3.22）
            from src.dashboard.custom_snapshots import load_snapshots
            snaps = load_snapshots(Path("data/custom_snapshots.csv"), sid)
            st.markdown(f"**{_label(sid)} 每日技術快照紀錄**"
                        "（方法論檢核值，不含模型數字）")
            if len(snaps) == 0:
                st.info("尚無快照紀錄。每日排程會對 repo 內 "
                        "custom_watchlist.csv 清單的股票自動累積技術快照——"
                        "把此代號寫入該檔並 push，明天起自動累積；"
                        "面板臨時加入的自選股不在排程範圍。")
            else:
                _sh = snaps.copy()
                _sh["last_bar_date"] = _sh["last_bar_date"].dt.strftime(
                    "%Y-%m-%d")
                _sh["bias"] = (_sh["bias"] * 100).round(1).astype(str) + "%"
                _sh = _sh[["last_bar_date", "close", "tidal", "bias",
                           "wave"]]
                _sh.columns = ["資料日", "收盤", "潮汐狀態", "量能乖離",
                               "波浪"]
                st.dataframe(_sh.tail(30), hide_index=True)
                st.caption("由每日排程自動累積（同日去重）；"
                           "完整歷史在 repo 的 data/custom_snapshots.csv。")
        hist = (load_predictions_view(P3.predictions_csv, sid)
                if not _sid_custom else
                load_predictions_view(P3.predictions_csv, "__none__"))
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

        elif sid and not _sid_custom:
            st.caption(f"{format_choice(sid, names)} 目前僅 {len(hist)} 筆紀錄，"
                       "累積 2 筆以上才顯示趨勢圖。")

        # ── 📐 預測準確度檢驗（前瞻，v3.21；與 7/30 裁決同一工具）──
        with st.expander("📐 預測準確度檢驗（前瞻已到期樣本）"):
            from src.dashboard.prospective_eval import (evaluate_from_frames,
                                                        summarize_accuracy)
            _cost = P1.fee_buy_rate + P1.fee_sell_rate + P1.tax_sell_rate
            _pred_all = load_predictions_view(P3.predictions_csv)
            _pred_top = _pred_all[_pred_all["stock_id"].isin(holding_ids)]
            _frames = {s: _view(s, pd.Timestamp.today().strftime("%Y-%m-%d")
                                )["ohlcv"] for s in
                       _pred_top["stock_id"].unique()}
            _ev = evaluate_from_frames(_pred_top, _frames, _cost,
                                       P2.forward_return_days)
            _sm = summarize_accuracy(_ev, P2.forward_return_days)
            st.markdown(f"**裁決狀態**：{_sm['verdict']}")
            if _sm["n_matured"] > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("已到期樣本", _sm["n_matured"])
                c2.metric("命中率", f"{_sm['hit_rate']:.1%}")
                c3.metric("二項檢定 p", f"{_sm['p_binom']:.3f}")
                _mt = _ev[_ev["matured"] == True].copy()   # noqa: E712
                _mt["last_bar_date"] = _mt["last_bar_date"].dt.strftime(
                    "%Y-%m-%d")
                _mt["realized_net"] = (_mt["realized_net"] * 100
                                       ).round(2).astype(str) + "%"
                _mt["proba_up"] = (_mt["proba_up"] * 100
                                   ).round(1).astype(str) + "%"
                _mt["stock_id"] = _mt["stock_id"].map(
                    lambda x: format_choice(x, names))
                show_ev = _mt[["stock_id", "last_bar_date", "proba_up",
                               "pick", "realized_net", "hit"]]
                show_ev.columns = ["代碼", "預測日", "P(漲)", "模型看多",
                                   "實際淨報酬", "命中"]
                st.dataframe(show_ev, hide_index=True)
            st.caption("命中定義：模型方向（P>50%＝看多）與實際 5 日淨報酬"
                       "正負一致；淨報酬＝毛報酬−總成本率（一階近似）。"
                       "凍結預測不重算；此檢驗即 Model v1 最終裁決之工具。")

