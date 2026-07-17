# -*- coding: utf-8 -*-
"""自選清單持久化（v3.11）：清單序列化進網址參數，使用者以書籤保存。

公開面板無帳號系統，伺服器端儲存會讓所有訪客共用清單（錯誤設計）；
網址參數是每使用者、零伺服器狀態的正確解。
"""
from __future__ import annotations


def parse_watch_param(s: str) -> list[str]:
    """網址參數 → 代號清單（僅收 4–6 位數字，去重保序，上限 30）。"""
    out: list[str] = []
    for tok in str(s or "").split(","):
        c = tok.strip()
        if c.isdigit() and 4 <= len(c) <= 6 and c not in out:
            out.append(c)
    return out[:30]


def serialize_watch(ids: list[str]) -> str:
    """代號清單 → 網址參數字串（去重保序）。"""
    return ",".join(dict.fromkeys(ids))
