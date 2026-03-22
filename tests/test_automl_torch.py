"""Optuna AutoML 與 Torch 主線整合（無依賴時略過）。"""
import numpy as np
import pytest


def test_run_optuna_automl_smoke():
    pytest.importorskip("optuna")
    from automl.optuna_runner import run_optuna_automl

    rng = np.random.default_rng(2)
    X = rng.normal(size=(45, 4)).astype(float)
    y = (X.sum(axis=1) + 0.05 * rng.normal(size=45)).astype(float)
    model, name, meta = run_optuna_automl(
        X,
        y,
        n_trials=4,
        include_deep=False,
        normalize=True,
        random_state=0,
    )
    assert model is not None
    assert name in ("linear", "xgboost", "lightgbm")
    assert "best_params" in meta


def test_unified_predict_lstm_smoke():
    pytest.importorskip("torch")
    from unified_predict import UnifiedPredictor

    rng = np.random.default_rng(3)
    X = rng.normal(size=(35, 5)).astype(float)
    y = (X[:, 0] * 0.5 + X[:, 1]).astype(float)
    p = UnifiedPredictor(auto_onnx=False, normalize=False, rate_limit_enabled=False)
    m = p.fit(
        X,
        y,
        model="lstm",
        seq_len=8,
        torch_epochs=5,
        torch_batch_size=16,
        early_stopping=False,
    )
    assert m["model"] == "lstm"
    ev = p.evaluate(X, y)
    assert ev["rmse"] >= 0
    assert p._torch_feature_attr is not None
    assert p._torch_feature_attr.shape[0] == 5
    assert abs(float(p._torch_feature_attr.sum()) - 1.0) < 1e-5


def test_predict_interval_naive():
    from unified_predict import UnifiedPredictor

    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3)).astype(float)
    y = X[:, 0]
    p = UnifiedPredictor(auto_onnx=False, normalize=False, rate_limit_enabled=False)
    p.fit(X, y, model="linear")
    out = p.predict_interval_naive(X[:4], z_score=1.0)
    assert "lower" in out and "upper" in out
    assert len(out["prediction"]) == 4
