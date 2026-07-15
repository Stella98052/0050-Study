# 流程自動化說明（GitHub Actions）

repo：https://github.com/Stella98052/0050-Study

## 你的 repo 現況（2026-07-15 檢查）與待補

| 項目 | 現況 | 待辦 |
|---|---|---|
| 程式碼（src/config/tests…） | ✅ 已上傳 | — |
| `.github/workflows/`（自動化）| ❌ 尚未上傳 | 本 pack 補上 |
| `.streamlit/`、`bootstrap_cloud.py`、`DEPLOY.md` | ❌ 尚未上傳 | 本 pack 補上 |
| `__pycache__/`（誤入版控）| ⚠ 已被 commit | setup_github.sh 清除 |
| `data/models/` 模型包 | ❓ 無法從外部確認 | setup 會自建並 commit |

## 一鍵設定（推薦）

解壓本 pack 覆蓋到你的本機 repo 後，在 repo 根目錄執行：

```
bash setup_github.sh
```

它會依序：清 __pycache__ → 確認自動化檔案 → 建置模型包（若無）→
跑 105 項測試 → 加入並 push。完成後照畫面指示啟用 Actions 與部署面板。

> Windows（無 bash）可用 Git Bash 執行，或手動照 setup_github.sh 裡的
> 五個步驟逐行貼到終端。

## 兩條自動化流水線（已就緒）

### 1. 每日前瞻更新 `daily_update.yml`
- **觸發**：台北時間每交易日 22:00（UTC 14:00）自動；也可手動 Run workflow
- **動作**：無模型包則先 run_phase2 自建 → 跑 daily_update（VG-1+預測）
  → 把 predictions.csv／模型包／快取 commit 回 repo
- **效果**：前瞻樣本每日累積，面板進度條自動前進；達 30 獨立樣本後
  才對 Model v2 裁決（預先宣告規則，不可提前偷看）
- **防護**：concurrency 防手動+排程撞車；timeout 40 分給足抓取時間

### 2. 測試 CI `tests.yml`
- **觸發**：每次 push / PR
- **動作**：跑 105 項 pytest；不綠顯示紅叉
- **意義**：治理鐵則（測試全綠才部署）延伸到雲端，防壞版本上線

## 自動化的邊界（誠實聲明）

- 自動化**只累積前瞻紀錄與更新資料**，不會改變科學結論——模型
  AUC≈0.5 無判別力這件事，每日更新不會讓它變好，只是誠實地累積
  未來真實資料等待最終裁決
- 自動化**不自動重訓擴充模型**。若要加自選股擴充訓練池（先前討論的
  方案 A/B），那是需要你決策的獨立步驟，不會被每日排程偷偷觸發——
  訓練池變更必須經過明確的 holdings 更新與 VG 驗證，這是治理要求
