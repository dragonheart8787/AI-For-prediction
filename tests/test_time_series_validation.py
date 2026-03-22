"""時間序列切分工具測試。"""
import numpy as np
import pytest

from validation.time_series_split import (
    last_fold_train_test,
    train_val_split_indices,
    walk_forward_splits,
)


def test_train_val_split_order():
    tr, va = train_val_split_indices(100, val_ratio=0.2, purge_gap=2)
    assert len(tr) >= 1 and len(va) >= 1
    assert np.all(tr < va.min())


def test_train_val_split_raises_small_n():
    with pytest.raises(ValueError):
        train_val_split_indices(1)


def test_walk_forward_yields_increasing_or_valid():
    folds = list(walk_forward_splits(50, n_splits=3, test_size=5, min_train_size=8, purge_gap=0))
    assert len(folds) >= 1
    for tr, te in folds:
        assert len(tr) >= 1 and len(te) >= 1
        assert tr.max() < te.min()


def test_last_fold_train_test():
    tr, te = last_fold_train_test(40, test_size=6, purge_gap=1, min_train_size=5)
    assert len(te) >= 1
    assert len(tr) >= 1
