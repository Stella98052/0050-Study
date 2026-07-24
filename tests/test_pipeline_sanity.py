# -*- coding: utf-8 -*-
"""每日管線端到端 sanity（v3.24）——針對「全綠卻無資料」類故障的守門測試。

背景（L54/L55/L56）：曾三度出現 workflow 全綠但 predictions 未更新，
根因各為 git add 多路徑 fatal、以 2>/dev/null 抑制錯誤、無模型包時
靜默略過寫入。本檔把每個保護點變成自動化斷言。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WF = Path(".github/workflows/daily_update.yml")
WD = Path(".github/workflows/watchdog.yml")


def _wf_text() -> str:
    assert WF.exists(), "daily_update.yml 不存在"
    return WF.read_text(encoding="utf-8")


def test_no_silent_failure_swallowing_on_critical_steps():
    """關鍵步驟不得以 `|| echo` / `|| true` 吞掉失敗（靜默斷更之源）。"""
    bad = []
    for line in _wf_text().splitlines():
        s = line.strip()
        if ("run_phase2.py" in s or "daily_update.py" in s) and \
                ("|| echo" in s or "|| true" in s):
            bad.append(s)
    assert not bad, f"關鍵步驟仍吞掉失敗：{bad}"


def test_model_pack_existence_is_enforced_in_workflow():
    """workflow 必須在建置後驗證模型包存在，否則 daily_update 寫不出預測。"""
    t = _wf_text()
    assert "model_phase2-v1.joblib" in t, "workflow 未檢查模型包檔名"
    assert "::error::" in t or "exit 1" in t, "模型包缺失時未讓步驟失敗"


def test_git_add_uses_single_pathspec_with_guard():
    """git add 每行單一路徑且有存在性守衛（多路徑任一不存在即全不 stage）。"""
    t = _wf_text()
    for line in t.splitlines():
        s = line.strip()
        if s.startswith("git add -f "):
            args = [a for a in s[len("git add -f "):].split()
                    if not a.startswith("2>") and a not in ("||", "true")]
            assert len(args) == 1, f"git add 多路徑易靜默失敗：{s}"
    assert "if [ -f" in t and "if [ -d" in t, "缺少存在性守衛"


def test_staged_files_are_logged():
    """commit 前必須印出 staged 清單——否則綠燈無法佐證資料是否入庫。"""
    assert "已 stage 檔案" in _wf_text(), "workflow 未輸出 staged 清單"


def test_daily_update_aborts_without_model_pack():
    """daily_update 於無模型包時必須明確中止（return 非 0），不得靜默續行。"""
    src = Path("daily_update.py").read_text(encoding="utf-8")
    m = re.search(r"if model is None:(.{0,600})", src, re.S)
    assert m, "找不到無模型分支"
    body = m.group(1)
    assert "return 2" in body or "return 1" in body, \
        "無模型時未以非零退出（會造成全綠但零預測）"


def test_watchdog_is_data_driven_not_calendar():
    """watchdog 必須比對官方交易日，而非日曆今日（否則休市必誤報）。"""
    assert WD.exists()
    t = WD.read_text(encoding="utf-8")
    assert "watchdog_check.py" in t, "watchdog 未使用資料驅動檢查腳本"
    assert Path("scripts/watchdog_check.py").exists()


def test_gitignore_and_force_add_consistency():
    """被 .gitignore 排除的產物，workflow 必須以 -f 強制加入（否則永不入庫）。"""
    gi = Path(".gitignore").read_text(encoding="utf-8")
    t = _wf_text()
    for item, pat in (("data/predictions.csv", "data/predictions.csv"),
                      ("data/models", "data/models")):
        if any(line.strip().startswith(item.split("/")[0] + "/")
               or item in line for line in gi.splitlines()):
            assert f'git add -f "$f"' in t or f'git add -f "$d"' in t, \
                f"{item} 受 gitignore 排除但 workflow 未強制加入"
