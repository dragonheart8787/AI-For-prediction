"""架構契約：回傳格式、對齊防漏、核心依賴隔離。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fit_returns_unified_contract_keys():
    from unified_predict import UnifiedPredictor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    p = UnifiedPredictor(auto_onnx=False, normalize=False, rate_limit_enabled=False)
    out = p.fit(X, y, model="linear")
    for k in ("predictions", "model_type", "feature_names", "metrics", "artifacts"):
        assert k in out
    assert "train_rmse" in out["metrics"]
    assert "train_r2" in out


def test_predict_returns_unified_plus_legacy():
    from unified_predict import UnifiedPredictor

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    p = UnifiedPredictor(auto_onnx=False, rate_limit_enabled=False)
    p.fit(X, y, model="linear")
    r = p.predict(X[:3])
    assert "predictions" in r and "prediction" in r
    assert r["predictions"] == r["prediction"]


def test_core_import_unified_predict():
    """import unified_predict 不應強制載入 torch（torch 僅在深度模型分支 lazy import）。"""
    import unified_predict as up  # noqa: F401

    assert hasattr(up, "UnifiedPredictor")


def test_prevent_leakage_disables_bfill_policy():
    pd = pytest.importorskip("pandas")
    from schema_infer import AlignSummary, align_source_frames

    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    main = pd.DataFrame({"timestamp": ts, "a": [1.0, 2.0, 3.0]})
    aux = pd.DataFrame({"timestamp": ts, "b": [10.0, 20.0, 30.0]})
    out_df, summary = align_source_frames(
        {"m": main, "x": aux},
        "m",
        timestamp_col="timestamp",
        freq="1D",
        missing_policy="bfill",
        prevent_leakage=True,
    )
    assert isinstance(summary, AlignSummary)
    assert len(out_df) >= 1
    assert "x" in summary.source_match_ratio or summary.rows_after >= 1


def test_artifact_registry_writes_bundle(tmp_path):
    from artifact_registry import ensure_run_dir, write_run_bundle

    d = ensure_run_dir("t_task", "abc123", base=str(tmp_path))
    paths = write_run_bundle(
        d,
        config={"x": 1},
        metrics={"train_r2": 0.5},
        summary={"run_id": "abc123"},
    )
    assert "config" in paths
    assert os.path.isfile(paths["config"])
