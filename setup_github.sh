#!/usr/bin/env bash
# 一鍵設定 GitHub 自動化（在 repo 根目錄執行一次）
# 用途：把自動化檔案就位、清掉不該進版控的暫存、建置模型包、推上 GitHub
set -e

echo "== 步驟 1/5：清除不該進版控的暫存檔 =="
git rm -r --cached __pycache__ 2>/dev/null || true
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "  已移除 __pycache__（.gitignore 之後會持續忽略）"

echo "== 步驟 2/5：確認自動化檔案就位 =="
for f in .github/workflows/daily_update.yml .github/workflows/tests.yml \
         .streamlit/config.toml packages.txt bootstrap_cloud.py \
         DEPLOY.md .gitignore; do
  [ -f "$f" ] && echo "  ✓ $f" || echo "  ✗ 缺 $f（請確認已解壓部署 pack）"
done

echo "== 步驟 3/5：本機建置模型包與快取（若尚未建置）=="
if [ ! -f data/models/model_phase2-v1.joblib ]; then
  echo "  建置中（首次約需數分鐘，含 TWSE 抓取）…"
  python run_phase2.py --holdings holdings.csv --skip-vg2
else
  echo "  模型包已存在，略過"
fi

echo "== 步驟 4/5：測試全綠才可部署（治理鐵則）=="
python -m pytest tests/ -q

echo "== 步驟 5/5：加入並推上 GitHub =="
git add .github .streamlit packages.txt bootstrap_cloud.py DEPLOY.md \
        .gitignore requirements.txt app_streamlit.py
git add -f data/models/model_phase2-v1.joblib data/reports/phase2_report.json
git add -f data/raw 2>/dev/null || true
git commit -m "feat: 雲端部署 + GitHub Actions 每日自動化 + 模型包/快取"
git push
echo ""
echo "✅ 完成。接下來："
echo "  1. GitHub repo → Actions 分頁 → 若提示啟用 workflow 點啟用"
echo "  2. https://share.streamlit.io 部署面板（詳見 DEPLOY.md）"
echo "  3. Actions → 每日前瞻更新 → Run workflow 可立即手動測試一次"
