#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多領域回測示範：爬取／合併資料 → 訓練 → 驗證指標。
每行輸出一個 JSON（供 CLI 整合測試與腳本解析）。

設定環境變數 PREDICT_AI_DEMO_FAST=1 時略過外部連線，僅用合成資料（供 CI／測試）。
"""
import argparse
import json
import os
import numpy as np

from data_connectors.yahoo import YahooFinanceConnector
from data_connectors.eia import EIAConnector
from data_connectors.newsapi import NewsAPIConnector
from schema_infer import infer_timestamp_key, rows_to_features
from unified_predict import UnifiedPredictor
from validation.time_series_split import train_val_split_indices


def to_matrix(rows):
    keys = sorted({k for r in rows for k in r.keys() if k != "timestamp"})

    def enc(v):
        try:
            return float(v)
        except Exception:
            return float(hash(str(v)) % 1000) / 1000.0

    X = np.asarray([[enc(r.get(k)) for k in keys] for r in rows], dtype=float)
    return X, keys


# (domain, horizon) 元資料列 — 指標共用同一 hold-out，僅標示領域／地平線
DOMAIN_HORIZONS = [
    ("financial", 1),
    ("financial", 5),
    ("weather", 1),
    ("energy", 1),
    ("medical", 1),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多領域回測示範：輸出每組 domain/horizon 的 JSON 指標行",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "xgboost", "lightgbm", "linear"],
        help="模型或 auto（依序嘗試 xgboost→lightgbm→linear）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4096,
        help="predict_many 批量大小",
    )
    args = parser.parse_args()

    if os.environ.get("PREDICT_AI_DEMO_FAST", "").strip().lower() in ("1", "true", "yes"):
        rng = np.random.default_rng(42)
        n, d = 500, 12
        X = rng.normal(size=(n, d))
        w = rng.normal(size=d)
        y = X @ w + rng.normal(scale=0.2, size=n)
    else:
        y_rows = list(YahooFinanceConnector().fetch(symbol="AAPL"))
        e_rows = list(EIAConnector().fetch())
        n_rows = list(NewsAPIConnector().fetch())

        rows = y_rows[:200] + e_rows[:200] + n_rows[:200]
        ts_key = infer_timestamp_key(rows)
        rows = rows_to_features(rows, ts_key)

        if not rows:
            print(json.dumps({"ok": False, "error": "no_rows"}, ensure_ascii=False))
            return

        X, _keys = to_matrix(rows)
        w = np.random.randn(X.shape[1])
        y = X @ w + np.random.randn(X.shape[0]) * 0.2

    train_idx, val_idx = train_val_split_indices(len(X), val_ratio=0.2, purge_gap=1)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    p = UnifiedPredictor(auto_onnx=True, rate_limit_enabled=False)
    model_name = args.model
    if model_name == "auto":
        try:
            p.fit(X_train, y_train, model="xgboost")
        except Exception:
            try:
                p.fit(X_train, y_train, model="lightgbm")
            except Exception:
                p.fit(X_train, y_train, model="linear")
    else:
        p.fit(X_train, y_train, model=model_name)

    batch = max(1, min(int(args.batch), X.shape[0]))
    out = p.predict_many(X, domain="custom", batch_size=batch)
    model_name_out = str(out.get("model", "linear"))
    onnx_used = "onnx" in model_name_out.lower()

    ev = p.evaluate(X_val, y_val)
    rmse = float(ev["rmse"])
    mae = float(ev["mae"])
    r2 = float(ev["r2"])
    mse = float(rmse ** 2)

    for domain, horizon in DOMAIN_HORIZONS:
        print(
            json.dumps(
                {
                    "model": model_name_out,
                    "domain": domain,
                    "horizon": int(horizon),
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "batch": int(batch),
                    "onnx": bool(onnx_used),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
