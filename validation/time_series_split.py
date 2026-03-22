"""
時間序列切分：依時間順序、purge gap、walk-forward，降低標籤洩漏。
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import numpy as np


def train_val_split_indices(
    n_samples: int,
    val_ratio: float = 0.2,
    purge_gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    訓練索引為前段連續區間，驗證為最後一段；中間可隔 purge_gap 樣本避免重疊標籤洩漏。

    Args:
        n_samples: 總樣本數（假設已按時間排序）
        val_ratio: 驗證集比例
        purge_gap: 訓練結尾與驗證開頭之間排除的樣本數（例如預測 horizon>1 時）
    """
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples 必須 >= 2")
    val_ratio = float(min(0.95, max(0.05, val_ratio)))
    purge_gap = max(0, int(purge_gap))

    n_val = max(1, int(round(n_samples * val_ratio)))
    n_train = n_samples - n_val - purge_gap
    if n_train < 1:
        purge_gap = max(0, n_samples - n_val - 1)
        n_train = n_samples - n_val - purge_gap
    if n_train < 1:
        n_train = 1
        n_val = max(1, n_samples - n_train - purge_gap)

    train_idx = np.arange(0, n_train, dtype=int)
    val_start = n_train + purge_gap
    if val_start >= n_samples:
        val_start = n_samples - n_val
        train_idx = np.arange(0, max(1, val_start - purge_gap), dtype=int)
    val_idx = np.arange(val_start, n_samples, dtype=int)
    if len(val_idx) == 0:
        val_idx = np.array([n_samples - 1], dtype=int)
    return train_idx, val_idx


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 3,
    test_size: Optional[int] = None,
    min_train_size: int = 10,
    purge_gap: int = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Walk-forward：訓練集逐步變長，每次在最後保留 test_size 作為測試（時間在後）。

    Yields:
        (train_indices, test_indices)
    """
    n_samples = int(n_samples)
    n_splits = max(1, int(n_splits))
    min_train_size = max(1, int(min_train_size))
    purge_gap = max(0, int(purge_gap))

    if test_size is None:
        test_size = max(1, n_samples // (n_splits + 2))
    test_size = int(test_size)

    if n_samples < min_train_size + test_size + purge_gap:
        train_idx, val_idx = train_val_split_indices(
            n_samples, val_ratio=test_size / n_samples, purge_gap=purge_gap
        )
        yield train_idx, val_idx
        return

    for k in range(n_splits):
        test_end = n_samples - k * test_size
        test_start = test_end - test_size
        if test_start < min_train_size + purge_gap:
            break
        train_end = test_start - purge_gap
        if train_end < min_train_size:
            break
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def last_fold_train_test(
    n_samples: int,
    test_size: int,
    purge_gap: int = 0,
    min_train_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """取得時間上最近一窗（測試集在最尾端）的 walk-forward train/test 索引。"""
    folds: List[Tuple[np.ndarray, np.ndarray]] = list(
        walk_forward_splits(
            n_samples,
            n_splits=999,
            test_size=test_size,
            min_train_size=min_train_size,
            purge_gap=purge_gap,
        )
    )
    if not folds:
        return train_val_split_indices(n_samples, val_ratio=0.2, purge_gap=purge_gap)
    return folds[0]
