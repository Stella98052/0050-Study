# -*- coding: utf-8 -*-
"""VG 狀態小卡資料（規格：讓使用者一眼看出統計可信度，未通過不可省略）。"""
from __future__ import annotations
import json
from pathlib import Path

_GATE_NOTES = {
    "VG-1": "資料完整性（型別/連續/負值/重複/完整率）",
    "VG-2": "存活偏誤對照組（差距須標註偏誤，不得宣稱優勢）",
    "VG-3": "規則訊號統計顯著性（獨立樣本 permutation+bootstrap）",
    "VG-4": "樣本量門檻（統計獨立層 ≥30）",
    "VG-5": "無未來資訊洩漏（截斷重算斷言）",
    "VG-6": "模型輸出健康度（逐列分布+AUC 判別力）",
}


def load_vg_status(report_path: Path) -> list[dict]:
    """讀 phase2_report.json 的 vg_summary → 卡片資料。缺檔回傳明確占位
    （不得默默顯示全綠）。"""
    if not report_path.exists():
        return [{"gate": g, "passed": None,
                 "note": _GATE_NOTES[g] + "｜⚠ 尚未執行 run_phase2，無報告"}
                for g in _GATE_NOTES]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("vg_summary", {})
    cards = []
    for g, note in _GATE_NOTES.items():
        cards.append({"gate": g, "passed": summary.get(g),
                      "note": note})
    return cards


def vg6_blocking(report_path: Path) -> tuple[bool, str]:
    """VG-6 未通過 → 預測卡必須掛醒目警語（模型無判別力，僅供架構演示）。"""
    for c in load_vg_status(report_path):
        if c["gate"] == "VG-6":
            if c["passed"] is False:
                return True, ("VG-6 未通過：模型無判別力（holdout AUC≈0.5）。"
                              "下方預測值不具統計意義，僅供架構演示，"
                              "不得作為進出依據。")
            if c["passed"] is None:
                return True, "VG-6 狀態未知（無 phase2 報告），預測僅供演示。"
    return False, ""
