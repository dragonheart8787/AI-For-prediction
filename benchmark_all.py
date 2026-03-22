#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能力與效能基準測試
分析與測試預測 AI 系統的全部能力和性能
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = {"timestamp": datetime.now().isoformat(), "tests": [], "summary": {}}


def run_section(name: str, func, *args, **kwargs):
    """執行測試區段並記錄結果"""
    start = time.perf_counter()
    try:
        out = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        RESULTS["tests"].append({"name": name, "status": "ok", "elapsed_ms": round(elapsed * 1000, 2), "output": out})
        return out
    except Exception as e:
        elapsed = time.perf_counter() - start
        RESULTS["tests"].append({"name": name, "status": "fail", "elapsed_ms": round(elapsed * 1000, 2), "error": str(e)})
        raise


def bench_unified_predictor():
    """UnifiedPredictor 能力與效能"""
    from unified_predict import UnifiedPredictor

    rng = np.random.default_rng(42)
    X = rng.normal(size=(2000, 8))
    y = X @ rng.normal(size=8) + rng.normal(0, 0.2, 2000)

    # 訓練
    p = UnifiedPredictor(auto_onnx=False, normalize=True, rate_limit_enabled=False)
    t0 = time.perf_counter()
    metrics = p.fit(X[:1500], y[:1500], model="linear")
    train_ms = (time.perf_counter() - t0) * 1000

    # 評估
    ev = p.evaluate(X[1500:], y[1500:])

    # 單次預測延遲（限制次數以避開 rate limit）
    t0 = time.perf_counter()
    for _ in range(50):
        p.predict(X[:10])
    predict_50_ms = (time.perf_counter() - t0) * 1000

    # 批次預測
    t0 = time.perf_counter()
    r = p.predict_many(X[:5000] if X.shape[0] >= 5000 else X, batch_size=512)
    batch_ms = (time.perf_counter() - t0) * 1000

    return {
        "train_ms": round(train_ms, 2),
        "train_r2": round(metrics.get("train_r2", 0), 4),
        "eval_r2": round(ev.get("r2", 0), 4),
        "predict_50_calls_ms": round(predict_50_ms, 2),
        "predict_many_ms": round(batch_ms, 2),
        "samples_per_sec": round(len(r.get("prediction", [])) / (batch_ms / 1000), 1) if batch_ms > 0 else 0,
    }


def bench_prediction_advisor():
    """預測目標顧問能力"""
    from prediction_target_advisor import advise_for_prediction_target

    cases = ["我想預測股價", "台北氣溫", "黃金價格", "能源需求", "未知目標xyz"]
    results = []
    t0 = time.perf_counter()
    for c in cases:
        r = advise_for_prediction_target(c)
        results.append({"input": c, "task_id": r["task_id"], "matched": r["matched"]})
    elapsed = (time.perf_counter() - t0) * 1000
    return {"cases": len(cases), "total_ms": round(elapsed, 2), "avg_ms": round(elapsed / len(cases), 2), "results": results}


def bench_training_store():
    """TrainingDataStore 能力"""
    from training_data_store import TrainingDataStore, merge_with_replay

    store = TrainingDataStore(path="data/benchmark_memory.jsonl")
    store.clear()

    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(100, 5))
    y1 = rng.normal(size=100)

    t0 = time.perf_counter()
    store.add(X1, y1, "task_a", "financial", ["a", "b", "c", "d", "e"])
    add_ms = (time.perf_counter() - t0) * 1000

    X2 = rng.normal(size=(50, 4))
    y2 = rng.normal(size=50)
    store.add(X2, y2, "task_b", "weather", ["x", "y", "z", "w"])

    t0 = time.perf_counter()
    X_r, y_r, _ = store.get_replay_samples(n_samples=50)
    replay_ms = (time.perf_counter() - t0) * 1000

    X_merged, y_merged = merge_with_replay(X2, y2, store, "task_b", replay_ratio=0.3)

    return {"add_ms": round(add_ms, 2), "replay_ms": round(replay_ms, 2), "merged_samples": len(y_merged)}


def bench_universal_predictor():
    """UniversalPredictor 能力"""
    from universal_predictor import UniversalPredictor

    up = UniversalPredictor()
    info = up.list_loaded_models()

    X = np.random.randn(10, 5).astype(float)
    names = ["open", "high", "low", "close", "volume"]

    # 已訓練任務
    r1 = up.predict(X, "stock_price_next", names) if "stock_price_next" in info.get("task_models", []) else None

    # 未訓練任務
    r2 = up.predict(X, "unknown_task_xyz", names)

    return {
        "task_models": info.get("task_models", []),
        "has_universal": info.get("has_universal", False),
        "unknown_task_source": r2.get("source", ""),
        "unknown_task_confidence": r2.get("confidence", 0),
    }


def bench_crawler_pipeline():
    """爬蟲訓練管線能力（單任務）"""
    from crawler_train_pipeline import crawl_and_build_xy, load_config

    config = load_config()
    t0 = time.perf_counter()
    X, y, fn, ti = crawl_and_build_xy("stock_price_next", config)
    elapsed = (time.perf_counter() - t0) * 1000
    return {"crawl_ms": round(elapsed, 2), "samples": ti["samples"], "features": ti["features"]}


def bench_data_connectors():
    """各資料連接器能力"""
    from crawler_train_pipeline import get_connector

    connectors = ["yahoo", "eia", "newsapi", "open_meteo"]
    results = {}
    for name in connectors:
        t0 = time.perf_counter()
        try:
            c = get_connector(name)
            if name == "yahoo":
                rows = list(c.fetch(symbol="AAPL", period="1mo"))
            elif name == "eia":
                rows = list(c.fetch())
            elif name == "newsapi":
                rows = list(c.fetch())
            else:
                rows = list(c.fetch(latitude=25, longitude=121))
            elapsed = (time.perf_counter() - t0) * 1000
            results[name] = {"ok": True, "rows": len(rows), "ms": round(elapsed, 2)}
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)[:80]}
    return results


def main():
    print("=" * 60)
    print("預測 AI 全能力與效能基準測試")
    print("=" * 60)

    sections = [
        ("UnifiedPredictor", bench_unified_predictor),
        ("PredictionAdvisor", bench_prediction_advisor),
        ("TrainingDataStore", bench_training_store),
        ("UniversalPredictor", bench_universal_predictor),
        ("CrawlerPipeline", bench_crawler_pipeline),
        ("DataConnectors", bench_data_connectors),
    ]

    for name, func in sections:
        print(f"\n[{name}] ...")
        try:
            out = run_section(name, func)
            print(f"  OK - {json.dumps(out, ensure_ascii=False)[:120]}...")
        except Exception as e:
            print(f"  FAIL - {e}")

    # 摘要
    ok = sum(1 for t in RESULTS["tests"] if t["status"] == "ok")
    fail = len(RESULTS["tests"]) - ok
    RESULTS["summary"] = {"passed": ok, "failed": fail, "total": len(RESULTS["tests"])}

    print("\n" + "=" * 60)
    print(f"摘要: {ok} 通過, {fail} 失敗")
    print("=" * 60)

    report_path = "reports/benchmark_report.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print(f"\n報告已儲存至 {report_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
