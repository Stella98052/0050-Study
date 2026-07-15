# 公開部署指南 — Streamlit Community Cloud + GitHub Actions

> repo：https://github.com/Stella98052/0050-Study
> 本指南把面板變成公開網頁，並用 GitHub Actions 每交易日自動更新前瞻紀錄。

## 為什麼是 Streamlit Cloud（而非 Cloudflare Pages）

此系統後端是 Python（抓 TWSE、跑 LightGBM、VG 驗證），需要能執行 Python
的伺服器。Cloudflare Pages / GitHub Pages 只能託管靜態 HTML，跑不了本面板。
Streamlit Community Cloud 免費、原生支援 Streamlit、直接連 GitHub、
每次 push 自動重新部署——最契合。

## 一、部署前置（在本機做一次）

**關鍵：先產生模型包與資料快取，再 commit，否則雲端首次啟動會逾時失敗。**

```
# 1) 產生模型包 data/models/model_phase2-v1.joblib 與 data/raw 快取
python run_phase2.py --holdings holdings.csv

# 2) 確認測試全綠（治理鐵則）
python -m pytest tests/ -q

# 3) 把模型包與快取一併納入 git（預設 data/ 常被 .gitignore 忽略，需確認）
git add -f data/models/model_phase2-v1.joblib
git add -f data/raw
git add data/reports/phase2_report.json
git add .streamlit packages.txt requirements.txt app_streamlit.py bootstrap_cloud.py
git add .github/workflows DEPLOY.md
git commit -m "deploy: 面板雲端部署設定 + 模型包 + 資料快取"
git push
```

> 註：`data/raw` 十檔十年 CSV 有一定體積但遠低於 GitHub 100MB 單檔限制；
> 若不想 commit 快取，面板技術圖首次載入會較慢（現抓官方資料）。

## 二、在 Streamlit Cloud 部署（約 3 分鐘）

1. 開 https://share.streamlit.io ，用 GitHub 帳號登入授權
2. 點 **Create app** → **Deploy a public app from GitHub**
3. 填：
   - Repository：`Stella98052/0050-Study`
   - Branch：`main`
   - Main file path：`app_streamlit.py`
4. 點 **Deploy**。首次建置約 2–5 分鐘（安裝 requirements）
5. 完成後得到公開網址 `https://<自訂名稱>.streamlit.app`，手機電腦皆可開

若面板顯示「雲端尚未完成建置」，代表模型包沒 commit 成功——回一、步驟 3
用 `git add -f` 強制加入 `data/models/`。

## 三、自動化（GitHub Actions，已含在 repo）

### 每日前瞻更新 `.github/workflows/daily_update.yml`
- 台北時間每交易日 22:00（UTC 14:00）自動跑 `daily_update.py`
- 累積前瞻預測至 `data/predictions.csv` 並 commit 回 repo
- Streamlit Cloud 偵測到 push 後自動重新部署，面板進度條隨之更新
- 也可在 GitHub → Actions → 手動 **Run workflow** 立即觸發

**啟用步驟**：push 後到 GitHub repo → Actions 分頁 → 若提示啟用 workflow
點啟用即可。權限已在 workflow 內宣告（contents: write）。

### 測試 CI `.github/workflows/tests.yml`
- 每次 push / PR 自動跑 pytest（104 項）
- 治理延伸：測試不綠時 PR 會顯示紅叉，避免把壞掉的版本部署上線

## 四、公開後的注意事項（重要）

- **這是公開網頁**：任何人可開啟。頂部固定風險聲明、VG-3❌/VG-6❌ 誠實
  顯示、自選股「模型不適用」說明——全部保留，確保公開情境下不誤導他人
- **不含任何金鑰**：面板只讀 TWSE 官方公開資料，無 API key、無帳密，
  可安全公開。（Fugle 降級鏈的 key 僅從環境變數讀，雲端未設即不啟用）
- **模型結論不變**：公開部署不改變科學結論——模型 AUC≈0.5 無判別力，
  面板僅供研究/教育與方法論演示，不構成投資建議

## 五、更新流程

改程式 → `git push` → Streamlit Cloud 自動重新部署（1–2 分鐘）。
改模型（重訓）→ 本機跑 run_phase2.py → `git add -f data/models` → push。
