#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速載入已儲存的 UnifiedPredictor 模型並預測（不依爬蟲管線）。

範例
----
  python quick_predict.py --list
  python quick_predict.py --task stock_price_next --info
  python quick_predict.py --task stock_price_next --values 1.0,2,3,4,5
  python quick_predict.py --model models/task_stock_price_next.pkl --csv my_rows.csv
  echo [[1,2,3,4,5],[1,2,3,4,5]] | python quick_predict.py --task stock_price_next --json -

模型目錄：環境變數 PREDICT_AI_MODELS_DIR 或 --models-dir，預設 ./models
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime_paths import resolve_models_dir  # noqa: E402
from unified_predict import UnifiedPredictor  # noqa: E402


def _models_root(ns: argparse.Namespace) -> str:
    return str(resolve_models_dir(getattr(ns, "models_dir", None)))


def list_saved_models(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    names: List[str] = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".pkl"):
            continue
        if fn.startswith("task_") or fn == "universal_model.pkl":
            names.append(fn)
    return names


def resolve_pkl_path(ns: argparse.Namespace) -> str:
    if ns.model:
        p = os.path.abspath(ns.model)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到模型檔：{p}")
        return p
    root = _models_root(ns)
    if ns.universal:
        p = os.path.join(root, "universal_model.pkl")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到通用模型：{p}")
        return p
    if not ns.task:
        raise ValueError("請指定 --task、--model 或 --universal")
    tid = ns.task.strip()
    p = os.path.join(root, f"task_{tid}.pkl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到任務模型：{p}（目錄 {root}）")
    return p


def _row_all_float(row: Sequence[str]) -> bool:
    try:
        for x in row:
            float(x.strip())
        return True
    except (TypeError, ValueError):
        return False


def load_features_csv(path: str, feature_names: Optional[List[str]]) -> np.ndarray:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("CSV 為空")
    if _row_all_float(rows[0]):
        arr = np.array([[float(c) for c in r] for r in rows], dtype=float)
        return arr
    header = [h.strip() for h in rows[0]]
    body = rows[1:]
    if not body:
        raise ValueError("CSV 只有表頭、無資料列")
    if not feature_names:
        arr = np.array([[float(c) for c in r] for r in body], dtype=float)
        return arr
    col_index = {h: i for i, h in enumerate(header)}
    missing = [fn for fn in feature_names if fn not in col_index]
    if missing:
        raise ValueError(f"CSV 表頭缺少與模型對應的欄位：{missing}；表頭為 {header}")
    out: List[List[float]] = []
    for r in body:
        if len(r) < max(col_index[fn] for fn in feature_names) + 1:
            raise ValueError(f"資料列欄位不足：{r!r}")
        out.append([float(r[col_index[fn]]) for fn in feature_names])
    return np.array(out, dtype=float)


def load_features_json(raw: str, feature_names: Optional[List[str]]) -> np.ndarray:
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("JSON 須為陣列，或單一物件")
    first = data[0]
    if isinstance(first, (int, float)):
        arr = np.array(data, dtype=float).reshape(1, -1)
        return arr
    if isinstance(first, list):
        return np.array(data, dtype=float)
    if isinstance(first, dict):
        if not feature_names:
            keys = sorted(first.keys())
            return np.array([[float(first[k]) for k in keys]], dtype=float)
        return np.array(
            [[float(row.get(fn, 0.0)) for fn in feature_names] for row in data],
            dtype=float,
        )
    raise ValueError("無法解析的 JSON 結構")


def parse_values_arg(s: str, n_expected: Optional[int]) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--values 不可為空")
    row = [float(x) for x in parts]
    arr = np.array([row], dtype=float)
    if n_expected is not None and arr.shape[1] != n_expected:
        raise ValueError(
            f"特徵數不符：收到 {arr.shape[1]} 個，模型需要 {n_expected} 個"
        )
    return arr


def build_predictor() -> UnifiedPredictor:
    return UnifiedPredictor(
        auto_onnx=False,
        normalize=False,
        rate_limit_enabled=False,
    )


def cmd_info(predictor: UnifiedPredictor, path: str) -> Dict[str, Any]:
    fn = predictor._feature_names
    return {
        "model_path": path,
        "model_type": predictor.model_name,
        "input_dim": predictor._input_dim,
        "feature_names": list(fn) if fn else [],
        "normalize": predictor.normalize,
        "fit_metrics": predictor._fit_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="快速載入 models/task_<任務>.pkl 並預測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--models-dir", default=None, help="模型目錄（預設見 runtime_paths）")
    ap.add_argument("--task", "-t", default=None, help="任務 ID，對應 task_<id>.pkl")
    ap.add_argument("--model", "-m", default=None, help="直接指定 .pkl 路徑")
    ap.add_argument("--universal", action="store_true", help="使用 universal_model.pkl")
    ap.add_argument("--list", action="store_true", help="列出模型目錄內可載入的 pkl")
    ap.add_argument("--info", action="store_true", help="只顯示模型欄位維度與特徵名稱")
    ap.add_argument(
        "--domain",
        default="custom",
        help="傳給 predict() 的 domain（影響 horizons 預設，多數迴歸可略）",
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--values",
        default=None,
        help="單筆預測：逗號分隔數值，順序須與訓練特徵一致",
    )
    src.add_argument("--csv", default=None, help="CSV：無表頭則每列為特徵；有表頭則依欄位名對齊")
    src.add_argument(
        "--json",
        default=None,
        metavar="PATH_OR_DASH",
        help='JSON：檔案路徑，或 "-" 從 stdin 讀取（陣列的陣列或物件陣列）',
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="僅輸出 predictions JSON，不印模型摘要",
    )
    ns = ap.parse_args()

    root = _models_root(ns)
    if ns.list:
        found = list_saved_models(root)
        print(json.dumps({"models_dir": root, "pkls": found}, ensure_ascii=False, indent=2))
        if not found:
            sys.exit(1)
        return

    path = resolve_pkl_path(ns)
    p = build_predictor()
    p.load_model(path)

    n_exp = p._input_dim
    fn = p._feature_names

    if ns.info:
        print(json.dumps(cmd_info(p, path), ensure_ascii=False, indent=2, default=str))
        return

    if not (ns.values or ns.csv or ns.json):
        print(
            "未指定輸入。請使用 --values、--csv 或 --json；或先執行 --info 查看特徵順序。",
            file=sys.stderr,
        )
        ap.print_help()
        sys.exit(2)

    if ns.values:
        X = parse_values_arg(ns.values, n_exp)
    elif ns.csv:
        X = load_features_csv(ns.csv, fn)
    else:
        assert ns.json is not None
        if ns.json.strip() == "-":
            raw = sys.stdin.read()
        else:
            with open(ns.json, encoding="utf-8") as f:
                raw = f.read()
        X = load_features_json(raw, fn)

    if n_exp is not None and X.shape[1] != n_exp:
        raise SystemExit(
            f"特徵維度錯誤：輸入 {X.shape[1]} 欄，模型需要 {n_exp} 欄。"
            + (f" 特徵順序：{fn}" if fn else "")
        )

    if not ns.quiet:
        print(
            json.dumps(
                {
                    "loaded": path,
                    "model_type": p.model_name,
                    "feature_names": list(fn or []),
                    "n_samples": int(X.shape[0]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )

    out = p.predict(X, domain=ns.domain)
    slim = {
        "predictions": out.get("predictions"),
        "model_type": out.get("model_type"),
        "confidence": out.get("confidence"),
        "feature_names": out.get("feature_names"),
    }
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
