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


@pytest.mark.parametrize("pyfile", ["daily_update.py", "feature_signal_audit.py",
                                    "run_phase2.py", "fetch_custom.py"])
def test_no_function_local_import_shadowing_module_level(pyfile):
    """L57：函式內 import 若與模組層級同名，會使該名稱在整個函式成為
    區域變數，導致較早的引用 UnboundLocalError——且 py_compile 驗不出。

    此測試曾攔下的實例：daily_update.main() 內重複 import
    build_feature_matrix，造成每日更新崩潰、資料靜默斷更 3 天。
    """
    import ast
    p = Path(pyfile)
    if not p.exists():
        pytest.skip(f"{pyfile} 不存在")
    tree = ast.parse(p.read_text(encoding="utf-8"))
    module_names = {
        (a.asname or a.name).split(".")[0]
        for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names}
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    nm = (a.asname or a.name).split(".")[0]
                    if nm in module_names:
                        offenders.append(f"{pyfile}:{n.lineno} {fn.name}() "
                                         f"重複 import {nm}")
    assert not offenders, "函式內 import 遮蔽模組層級名稱：" + "; ".join(offenders)


def test_daily_update_writes_predictions_end_to_end(tmp_path, monkeypatch):
    """端到端：注入假抓取器與假模型，驗證 main() 真的寫出預測列。

    這是對「全綠卻無資料」最直接的守門——只要控制流有任何中斷
    （例外、提早 return、寫檔失敗），本測試即失敗。
    """
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, ".")
    import daily_update as du

    days = pd.bdate_range("2024-01-01", periods=300)

    def fake_fetch(sid, s, e, cfg, session=None):
        px = 100 + np.cumsum(np.random.RandomState(7).randn(len(days)))
        return pd.DataFrame({"stock_id": sid, "date": days, "open": px,
                             "high": px + 1, "low": px - 1, "close": px,
                             "volume": 1000 + np.arange(len(days))})

    monkeypatch.setattr(du, "fetch_stock_history", fake_fetch)
    monkeypatch.setattr(du, "load_model_pack", lambda p: (object(), {}))
    monkeypatch.setattr(du, "predict_latest", lambda m, b, f: {
        "proba_up": 0.55, "pick": True, "model_tag": "test",
        "as_of": str(pd.to_datetime(f["date"].iloc[-1]).date())})

    h = tmp_path / "h.csv"
    h.write_text("stock_id,weight\n2330,1.0\n", encoding="utf-8")
    out = tmp_path / "pred.csv"

    # Phase3Config 為 dataclass：需在建構後改實例屬性（改 class 無效）
    _orig_p3 = du.Phase3Config

    def _p3_with_tmp(*a, **k):
        cfg = _orig_p3(*a, **k)
        object.__setattr__(cfg, "predictions_csv", out)
        return cfg

    monkeypatch.setattr(du, "Phase3Config", _p3_with_tmp)
    monkeypatch.setattr(sys, "argv", ["daily_update.py", "--holdings", str(h)])

    code = du.main()
    assert code == 0, f"main() 非零退出（{code}）＝當日不會有任何預測"
    assert out.exists(), "predictions 未寫出（靜默斷更的典型症狀）"
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) >= 2
