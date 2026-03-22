import json
import os
import sys
import numpy as np

from unified_predict import UnifiedPredictor


def main() -> int:
    summary = {"export_ok": False, "load_ok": False, "errors": [], "hints": []}

    # 準備資料
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 8))
    y = X @ rng.normal(size=(8,)) + rng.normal(scale=0.1, size=(256,))

    p = UnifiedPredictor()
    p.fit(X[:128], y[:128], model="linear")
    os.makedirs("models", exist_ok=True)

    onnx_path = os.path.join("models", "linear.onnx")

    # 匯出 ONNX
    try:
        out_path = p.export_onnx(onnx_path, input_dim=8)
        summary["export_ok"] = os.path.exists(out_path)
    except Exception as e:
        summary["errors"].append(f"export_err: {e}")
        summary["hints"].append("安裝 skl2onnx: pip install skl2onnx")

    # 載入 ONNX 並預測
    try:
        p.load_onnx(onnx_path, intra_threads=0, inter_threads=0)
        out = p.predict(X[:4])
        summary["load_ok"] = True
        preds = out.get("prediction", [])
        if isinstance(preds, list) and preds:
            summary["onnx_prediction_shape"] = [len(preds), len(preds[0]) if isinstance(preds[0], list) else 1]
        else:
            summary["onnx_prediction_shape"] = [0, 0]
    except Exception as e:
        summary["errors"].append(f"load_err: {e}")
        summary["hints"].append("安裝 onnxruntime: pip install onnxruntime")

    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["export_ok"] and summary["load_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

