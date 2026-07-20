# -*- coding: utf-8 -*-
"""自選股資料儲存 CLI（v3.16）。

用法：
  python fetch_custom.py --symbols 6669,2337
  python fetch_custom.py --watchlist custom_watchlist.csv --with-valuation
產物：data/custom/{代號}_ohlcv.csv、{代號}_features.csv（無未來報酬欄）、
     （選）{代號}_valuation.csv。與模型池完全隔離。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.config import Config
from config.phase2_config import Phase2Config
from src.custom.fetch_and_save import load_watchlist, save_custom_stock

DISCLAIMER = ("【風險聲明】本系統僅供研究與教育用途，歷史回測績效不代表"
              "未來表現，不構成投資建議。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="",
                    help="逗號分隔代號（例：6669,2337）")
    ap.add_argument("--watchlist", type=Path,
                    default=Path("custom_watchlist.csv"),
                    help="凍結清單檔（stock_id 欄）")
    ap.add_argument("--with-valuation", action="store_true",
                    help="一併儲存官方估值（BWIBBU）")
    ap.add_argument("--outdir", type=Path, default=Path("data/custom"))
    args = ap.parse_args()
    print(DISCLAIMER)
    ids = [c.strip() for c in args.symbols.split(",") if c.strip().isdigit()]
    if not ids:
        ids = load_watchlist(args.watchlist)
    if not ids:
        print("未指定代號：--symbols 6669,2337 或建立 custom_watchlist.csv")
        return 1
    print(f"自選股資料儲存：{ids} → {args.outdir}（與模型池隔離）")
    p1, p2 = Config(), Phase2Config()
    for sid in dict.fromkeys(ids):
        r = save_custom_stock(sid, p1, p2, args.outdir,
                              with_valuation=args.with_valuation)
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {r['stock_id']}：{r['rows']} 根日K｜{r['msg']}")
    print("完成。產物僅供研究；自選股不入模型池（治理裁定）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
