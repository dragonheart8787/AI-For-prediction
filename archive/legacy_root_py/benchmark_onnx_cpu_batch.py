import os
import sys
import time
import json
import numpy as np

from unified_predict import UnifiedPredictor, xgb, convert_sklearn, FloatTensorType, ort


def ensure_requirements(summary):
    missing = []
    if convert_sklearn is None or FloatTensorType is None:
        missing.append("skl2onnx")
    if ort is None:
        missing.append("onnxruntime")
    if missing:
        summary["errors"].append("缺少依賴: " + ", ".join(missing))
        for m in missing:
            if m == "skl2onnx":
                summary["hints"].append("pip install skl2onnx")
            if m == "onnxruntime":
                summary["hints"].append("pip install onnxruntime")
        return False
    return True


def main() -> int:
    summary = {"ok": False, "errors": [], "hints": [], "results": []}
    if not ensure_requirements(summary):
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    rng = np.random.default_rng(0)
    n_train, n_feat = 8000, 32
    X = rng.normal(size=(n_train, n_feat)).astype(np.float32)
    w = rng.normal(size=(n_feat,)).astype(np.float32)
    y = (X @ w + rng.normal(scale=0.2, size=(n_train,))).astype(np.float32)

    p = UnifiedPredictor()
    model_name = "xgboost" if xgb is not None else "linear"
    p.fit(X, y, model=model_name)

    os.makedirs("models", exist_ok=True)
    onnx_path = os.path.join("models", f"{model_name}.onnx")

    try:
        p.export_onnx(onnx_path, input_dim=n_feat)
    except Exception as e:
        summary["errors"].append(f"export_err: {e}")
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    try:
        p.load_onnx(onnx_path, intra_threads=0, inter_threads=0)
    except Exception as e:
        summary["errors"].append(f"load_err: {e}")
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    # 基準測試批量大小
    batch_sizes = [2048, 4096, 8192]
    Xq = rng.normal(size=(max(batch_sizes), n_feat)).astype(np.float32)

    for bs in batch_sizes:
        # 預熱
        _ = p.predict_many(Xq[:bs], domain="custom", batch_size=bs)
        # 正式量測
        t0 = time.perf_counter()
        out = p.predict_many(Xq[:bs], domain="custom", batch_size=bs)
        t1 = time.perf_counter()
        dt = max(t1 - t0, 1e-9)
        qps = bs / dt
        summary["results"].append({
            "batch_size": bs,
            "time_sec": dt,
            "throughput_rows_per_sec": qps,
            "model": out.get("model"),
        })

    summary["ok"] = True
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
