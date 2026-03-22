"""Walk-forward 與實驗紀錄測試。"""
import json
from pathlib import Path

import numpy as np

from experiment_log import log_experiment
from validation.walk_forward_eval import walk_forward_model_eval


def test_walk_forward_model_eval_runs():
    rng = np.random.default_rng(0)
    n, d = 80, 4
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + rng.normal(scale=0.1, size=n)
    out = walk_forward_model_eval(
        X,
        y,
        model="linear",
        n_splits=3,
        test_size=10,
        min_train_size=20,
        purge_gap=1,
        normalize=True,
    )
    assert out["n_folds"] >= 1
    assert out["mean_rmse"] is not None
    assert len(out["folds"]) == out["n_folds"]


def test_walk_forward_automl_returns_error_not_runs():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    y = X[:, 0]
    out = walk_forward_model_eval(
        X,
        y,
        model="automl",
        n_splits=3,
        test_size=10,
        min_train_size=20,
    )
    assert out["n_folds"] == 0
    assert out.get("error")


def test_walk_forward_skips_deep_without_wf_allow_deep():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 3))
    y = X[:, 0]
    out = walk_forward_model_eval(
        X,
        y,
        model="lstm",
        n_splits=3,
        test_size=10,
        min_train_size=20,
        wf_allow_deep=False,
    )
    assert out["n_folds"] == 0
    assert "wf-allow-deep" in (out.get("error") or "").lower()


def test_experiment_log_jsonl(tmp_path: Path):
    p = tmp_path / "runs.jsonl"
    log_experiment("test_event", {"a": 1}, path=str(p))
    log_experiment("test_event2", {"b": 2}, path=str(p))
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    j0 = json.loads(lines[0])
    assert j0["event"] == "test_event"
    assert j0["a"] == 1
