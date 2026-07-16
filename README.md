# 0050 量化分析系統

**公開面板：** https://0050-study-lala.streamlit.app
**GitHub：** https://github.com/Stella98052/0050-Study

> **風險聲明**：本系統僅供研究與教育用途，歷史回測績效不代表未來表現，
> 不構成投資建議。波浪理論屬主觀分析框架，本系統採規則化近似
> （approximation），非唯一標準。

---

## 這個系統是什麼

針對台灣 50 ETF（0050）前十大持股的量化分析與預測系統，整合 Elliott 波浪
理論、MV 潮汐量能、七道驗證關卡（VG-1~VG-7），並以「零猜想治理」為原則：
除股票代號外，所有數值皆由官方市場資料經公式計算，每項輸出可追溯來源、
通過單元測試才交付。

**重要科學結論（誠實揭露）**：在目前的資料、特徵集、驗證設計與樣本條件下，
本系統的規則訊號與 LightGBM 模型對未來 5 日淨報酬**未觀察到可重現且可驗證
的預測優勢**（VG-3 p=0.22、模型 holdout AUC≈0.494 等同隨機）。這不是失敗——
驗證關卡正確攔下了「看似有效實則無效」的假訊號。面板據此如實顯示
VG-3❌/VG-6❌，並對預測數字標註「僅供架構演示、不得作為進出依據」。
詳見 `PHASE2_CONCLUSION.md`。

**外部效度限制（審查採納，2026/7/16）**：上述結論僅適用於本研究的條件組合
——0050 前十大股票池、本套特徵構造、5 日持有期、與樣本期內的台股市場
狀態。否定結果**不應**未經限定地推廣為「所有波浪/量能策略無效」；同理，
更換股票池、特徵或持有期後的任何新主張，也須重新通過同一套驗證關卡。

---

## 目錄

1. [快速開始（本機）](#快速開始本機)
2. [公開面板部署（Streamlit Cloud）](#公開面板部署streamlit-cloud)
3. [每日自動化（GitHub Actions）](#每日自動化github-actions)
4. [面板使用說明](#面板使用說明)
5. [方法論核心](#方法論核心)
6. [專案結構](#專案結構)
7. [常見問題與除錯](#常見問題與除錯)

---

## 快速開始（本機）

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                        # 109 項單元測試，應全過
python run_phase2.py --holdings holdings.csv      # 建置模型包（首次含 TWSE 抓取，數分鐘）
streamlit run app_streamlit.py                    # 啟動本機面板 → 自動開 http://localhost:8501
```

> 首次執行某股票會抓 TWSE 官方日K（逐月快取，禮貌延遲），較慢屬正常；
> 之後走快取。

### 其他常用指令
```bash
python run_phase1.py --holdings holdings.csv                 # 第一階段資料管線
python daily_update.py --holdings holdings.csv              # 每日更新+前瞻紀錄
python feature_signal_audit.py --holdings holdings.csv --v2 # 特徵訊號稽核
python diagnose_duplicates.py 2330                          # 快取異常診斷
```

---

## 公開面板部署（Streamlit Cloud）

讓面板變成公開網址，且**不必再開 CMD `streamlit run`**——部署一次後，
直接開網址即可，資料每日自動更新。

### 為什麼需要 Streamlit Cloud（而非只靠 GitHub）
- **GitHub Actions** 是「排程執行器」：每天定時跑一次程式、產生資料，跑完
  即結束，不提供可互動的網頁。
- **Streamlit Cloud** 是「網頁伺服器」：把面板持續掛在網址上，讓人用瀏覽器
  互動（切股票、看圖）。GitHub 本身沒有這個功能（GitHub Pages 只能架
  靜態 HTML，跑不了本系統的 Python 後端）。
- 兩者搭配：Actions 每天更新資料 → Cloud 讀資料顯示。不是二選一。

### 部署步驟
1. **確認程式碼已 push 到 GitHub**：`git status` 顯示 clean 且 up to date
2. **本機先建置並提交模型包與快取**（否則雲端首載會逾時失敗）：
   ```bash
   python run_phase2.py --holdings holdings.csv
   git add -f data/models/model_phase2-v1.joblib data/reports/phase2_report.json
   git add -f data/raw
   git commit -m "add: 模型包與資料快取"
   git push origin main
   ```
3. 開 https://share.streamlit.io ，用 GitHub 帳號登入授權
4. **Create app → Deploy a public app from GitHub**，填：
   - Repository：`Stella98052/0050-Study`
   - Branch：`main`
   - Main file path：`app_streamlit.py`
5. 按 **Deploy**，等 2–5 分鐘 → 得到公開網址

> **不要用本機面板右上角的「Deploy」按鈕**：那是 Streamlit 內建快捷鍵，
> 流程較繞且常因未連 GitHub 而失敗（會跳「Unable to deploy / not connected
> to a remote GitHub repository」）。一律改用上述 share.streamlit.io 網站部署。

### 部署後更新
- 改程式 → `git push` → Cloud 1–2 分鐘自動重新部署
- 改模型 → 本機 `run_phase2.py` → `git add -f data/models` → push
- 雲端讀不到最新 → Streamlit Cloud 頁面右上選單 **Reboot app** 強制重抓

---

## 每日自動化（GitHub Actions）

repo 內含兩條流水線（`.github/workflows/`），push 後到 repo 的 **Actions**
分頁啟用即可。

### 1. 每日前瞻更新 `daily_update.yml`
- **觸發**：台北時間每交易日 **20:00**（UTC 12:00）；也可手動 Run workflow
- **動作**：跑 `daily_update.py`（VG-1 檢查 + 各股預測）→ 把 `predictions.csv`
  累積並 commit 回 repo → 面板自動反映
- **為何 20:00**：台股 13:25 收盤，但盤後有集合競價、鉅額、盤後定價
  （14:00–14:30）等處理，官方 API 通常傍晚才穩定齊備且未保證時間。20:00
  為保守安全邊際（本系統累積前瞻樣本、非即時交易，資料正確優先於即時）。
  若過早抓到前一交易日資料，`daily_update` 會警示。

### 2. 單元測試 `tests.yml`
- **觸發**：每次 push / PR
- **動作**：跑 109 項 pytest；不綠顯示紅叉（治理鐵則：測試綠才部署）

### 自動化邊界（治理）
自動化**只累積前瞻紀錄與更新資料，不自動重訓或擴充訓練池**。訓練池變更
（如加自選股擴充樣本）須經明確 holdings 更新與 VG 驗證，不被排程觸發。

---

## 面板使用說明

### 兩類資訊分區（重要）
- **本股即時值（隨股票變動）**：預測卡的方向、P(未來5日淨報酬>0)、
  realtime 波浪位置、13MV 核心否決線——切換股票會變。
- **全模型層級（每檔相同）**：VG-1~VG-6 驗證關卡——十檔以同一模型訓練，
  每檔顯示相同是設計、非錯誤。

### 自選股票
側欄「自選股票」輸入任意台股代號（0050 以外亦可，如 2337）按 Enter 加入。
- ✅ 有效：K線、波浪轉折、MV 潮汐（走官方資料，與前十大同一引擎）
- ❌ 不適用：模型預測（模型僅以 0050 前十大訓練，未見過自選股，故隱藏預測卡）

### 每日前瞻紀錄
面板自動讀取 Actions 累積的 `predictions.csv`，顯示各股最新預測總覽表 +
選定股票的預測歷史趨勢圖。VG-6 現況下模型無判別力，僅供前瞻管線演示。

---

## 方法論核心

- **Elliott 波浪**：5 推動 3 調整；規則化近似，非唯一標準
- **MV 潮汐量能**：5MV 短線、13MV 中線核心否決線（下彎＝絕對否決，凌駕
  一切訊號）、20MV 乖離率分母。方向比數值重要。
- **七道驗證關卡**：VG-1 資料完整性｜VG-2 存活偏誤對照｜VG-3 統計顯著性
  （獨立樣本 permutation+bootstrap）｜VG-4 樣本量門檻｜VG-5 防未來洩漏
  （截斷重算斷言）｜VG-6 模型輸出健康度（AUC 判別力）｜VG-7 特徵真偽篩
  （排除離散度×固定門檻機械假象）
- 公式細節見 `FORMULAS.md`；開發教訓見 `LESSONLEARNT.md`

---

## 專案結構

```
config/          全域參數（Config / Phase2Config / Phase3Config）
src/fetch/       TWSE 官方抓取（逐月快取+重試+越月防護）、持股清單、產業分類
src/wave/        ZigZag 轉折、三大鐵律波浪標籤
src/volume/      MV 5/13/20 潮汐三線
src/features/    技術指標、特徵矩陣（含防洩漏斷言）
src/model/       LightGBM 訓練、Walk-Forward、績效指標、序列化
src/validate/    VG-1~VG-7 七道驗證關卡
src/dashboard/   面板資料/圖表/模型/VG卡/前瞻紀錄（純函式，可測）
tests/           109 項 pytest（含環境自檢 test_environment.py）
app_streamlit.py 面板主程式
run_phase1/2.py  各階段主流程
daily_update.py  每日更新+前瞻紀錄
```

---

## 常見問題與除錯

**面板顯示「模型包不存在」**
→ `data/models/` 沒 commit。本機 `run_phase2.py` 後 `git add -f data/models` 再 push。

**面板技術圖出不來或很慢**
→ `data/raw` 快取沒 commit。`git add -f data/raw` 後 push。

**Streamlit Cloud 部署失敗「Unable to locate package …」**
→ `packages.txt` 含註解。此檔走 apt，**不支援 # 註解**，只能一行一個純套件名。
   確認內容只有 `libgomp1`：`printf 'libgomp1\n' > packages.txt` 後 push。
   確認 GitHub 上實際內容：`git show origin/main:packages.txt`

**CI「單元測試」紅叉「scikit-learn is required」**
→ `requirements.txt` 缺 scikit-learn（LightGBM 的 LGBMClassifier 需要）。
   確認含 `scikit-learn>=1.3`。`test_environment.py` 會在缺相依時最先失敗並點名。

**git「not a git repository」**
→ 在錯的目錄。先 `cd "~/OneDrive/AI/Claude/0050 Study/phase1"` 進 repo 再操作。

**面板右上「Deploy」按鈕跳 Unable to deploy**
→ 別用該按鈕。改用 share.streamlit.io 網站部署（見上方部署步驟）。

**Windows 貼指令出現 `app\_streamlit.py` 找不到檔案**
→ 從 Markdown 複製帶入了跳脫字元 `\_`。底線前不要有反斜線：`app_streamlit.py`。

**Streamlit Cloud 改了檔案但沒生效**
→ push 後 Cloud 有快取。到 Cloud 頁面右上選單 **Reboot app** 強制重抓。
