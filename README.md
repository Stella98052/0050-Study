# 0050 量化分析系統 — 第一階段（v1.6 / 2026-07-12 收官版）

【風險聲明】本系統僅供研究與教育用途，歷史回測績效不代表未來表現，不構成投資建議。
波浪理論屬主觀分析框架，本系統採規則化近似（approximation），非唯一標準。

## 專案檔案結構
```
phase1/
├── config/config.py            全域 Config（唯一參數來源）+ VERSION + 風險聲明
├── data/raw/{stock_id}/{yyyymm}.csv    逐月快取（當月每次重抓覆寫）
├── data/processed/             特徵輸出 {stock_id}_features.csv
├── data/validation_report_{id}.json    VG-1 報告
├── logs/fetch_failures.log     抓取失敗記錄
├── src/
│   ├── schemas.py              OHLCV_SCHEMA / Pivot / WaveSegment / VG1Report...
│   ├── fetch/twse_daily.py     TWSE 日K（逐月+快取+重試+越月防護）
│   ├── fetch/holdings.py       0050 持股清單（白名單+CSV fallback，禁捏造）
│   ├── fetch/auto_fetcher.py   多來源層：TWSE/Fugle 轉接器+降級鏈
│   ├── validate/trading_calendar.py  聯集日曆+官方休市日輔助校驗（定案1）
│   ├── validate/vg1.py         VG-1 完整性關卡
│   ├── wave/zigzag.py          ZigZag 轉折（retrospective/realtime）
│   ├── wave/wave_labels.py     三大鐵律+波浪標籤（定案3）
│   ├── volume/volume_features.py     MV 5/13/20 三線（定案2）+量價背離
│   └── signal_events.py        第3浪+潮汐爆發訊號+雙軌計數（定案4）
├── tests/                      52 項 pytest（防前視/重現性/VG-1/防護/雙軌）
├── run_phase1.py               主流程：清單→抓取→VG-1→波浪→量能→訊號
├── auto_update.py              每日收盤後自動更新（掛工作排程器）
├── diagnose_duplicates.py      快取污染診斷工具
├── FORMULAS.md / LESSONLEARNT.md / README.md
```

## 執行
```
pip install pandas numpy requests pytest
python -m pytest tests/ -v                       # 52 項單元測試
python run_phase1.py --holdings holdings.csv     # 首次完整建置
python auto_update.py --holdings holdings.csv    # 每日收盤後（建議排程 15:00 後）
python diagnose_duplicates.py 2330               # 懷疑快取異常時
```
啟動第一行必見「▶ 0050 phase1 v1.6」版本橫幅；不符即為版本混用。

## 核心函式：輸入/輸出/用途（★=可安全用於模型訓練 ◇=僅供視覺化）
| 函式 | 輸入 → 輸出 | 用途 | 標記 |
|---|---|---|---|
| fetch_stock_history | stock_id, start, end → OHLCV DataFrame | 逐月抓取+快取+越月防護+去重 | ★ |
| fetch_0050_top10 / load_holdings_from_csv | → HoldingsSnapshot | 持股清單（白名單/人工覆寫） | — |
| build_union_calendar | {id: df} → DatetimeIndex | 理論交易日曆（聯集，定案1） | — |
| fetch_twse_holidays / cross_check_calendar | → set[date] / warnings | 官方休市表+涵蓋範圍內校驗 | — |
| run_vg1_validation | df, calendar → VG1Report | 完整性關卡（未過必警示） | — |
| compute_pivots_retrospective | df → list[Pivot] | 回溯轉折（含暫定極值） | ◇ |
| compute_pivots_realtime | df, as_of → list[Pivot] | 已確認轉折（竄改未來不變） | ★ |
| label_waves_retrospective | df → list[WaveSegment] | 回溯標籤（圖表用） | ◇ |
| label_waves_realtime | df → 逐日標籤 DataFrame | basis_date≤date 逐列保證 | ★ |
| compute_mv_features | df → mv_short/mid/long+方向+bias+veto | 5/13/20 三線（定案2） | ★ |
| detect_price_volume_divergence | df, mv, labels → bool Series | 波3量價背離（防未來版） | ★ |
| detect_wave3_tidal_burst | labels, mv, div → bool Series | 訊號（含13MV否決） | ★ |
| extract_signal_events / count_statistically_independent_signals | → 事件/獨立樣本 | 定案4雙軌；VG-3/4 只用後者 | ★ |

VG-5 前置：◇ 函式輸出禁入訓練/驗證/參數選擇；第二階段將以管線斷言強制。

## 定案對照（四項全落實）
定案1 聯集日曆+官方休市輔助（涵蓋範圍內）｜定案2 三線+13MV核心否決（訊號
一票否決）｜定案3 鐵律二 partial/final 雙 key｜定案4 事件層/統計獨立層雙軌

## 已知限制
聯集日曆無法偵測「全體同日缺漏」（颱風臨時休市即此類→人工查證，見
LESSONLEARNT L4）｜持股清單來源不穩需 CSV 覆寫｜realtime 標籤天然延遲
（轉折需反向5%確認）｜鐵律二波5未現前為暫定判斷

## 驗收狀態（2026-07-12）
UT 52/52 ✅｜VG-1 十檔全過 ✅｜實戰異常攔截：TWSE回錯月✅、颱風臨時休市
標記✅、當月快取凍結修復✅｜第二階段前置（訊號樣本數預覽）：達30門檻
7檔；2454(25)/3711(24)/2317(20) 不足將如實標註

---
# 第二階段（v2.0 / 2026-07-12）

## 新增檔案
```
config/phase2_config.py            Phase2Config + PHASE2_VERSION
src/features/tech_indicators.py    RSI14 / MACD(12,26,9)
src/features/feature_matrix.py     特徵矩陣+標籤（⛔內建VG-5斷言）
src/model/train.py                 LightGBM（固定種子、scale_pos_weight）
src/model/walk_forward.py          切分（3y/3m/3m/embargo30/holdout12m）+逐折回測
src/model/metrics.py               Sharpe/MDD/勝率/盈虧比/Alpha（gross/net並列）
src/model/sensitivity.py           ZigZag 3%–8% 敏感度+熱圖（樣本內標註）
src/model/serialize.py             joblib 模型包（特徵清單/Config/VG摘要/版本）
src/validate/vg5_asserts.py        防洩漏斷言（黑名單/截斷重算/切分）
src/validate/vg2_survivorship.py   對照組（隨機10檔上市，固定種子）
src/validate/vg3_significance.py   permutation + bootstrap
src/validate/vg4_sample.py         樣本量關卡（沿用定案4獨立計數）
run_phase2.py                      主流程+VG-1~5總結報告
tests/test_phase2_core.py          13項測試
```

## 執行
```
pip install -r requirements.txt
python run_phase2.py --holdings holdings.csv                # 主流程（首跑含0050基準抓取）
python run_phase2.py --holdings holdings.csv --sensitivity  # 加跑敏感度（6閾值×全流程，耗時）
python run_phase2.py --holdings holdings.csv --skip-vg2     # 跳過對照組（省10檔×10年抓取）
```
輸出：data/reports/phase2_report.json（VG-1~5逐項狀態，未通過不省略）、
data/models/model_phase2-v1.joblib（第三階段載入）。

## 注意
- VG-2 首跑需抓對照組10檔×10年（約20–25分鐘，之後走快取）
- 敏感度分析為樣本內參數優化（報告明確標註）；最終評估以 holdout 為準
- VG-4 預告：2454/3711/2317 獨立訊號<30，統計結論將如實標「不可靠」
