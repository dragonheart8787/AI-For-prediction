#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型可解釋性（與推論核心分離）：Torch 梯度近似特徵重要性等。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def torch_gradient_feature_vector(
    model: Any,
    X_arr: np.ndarray,
    *,
    max_samples: int = 64,
) -> Optional[np.ndarray]:
    """
    對 ``TorchRegressorWrapper`` 計算輸入梯度近似重要性向量（已正規化合 1）。
    失敗回傳 None。
    """
    if type(model).__name__ != "TorchRegressorWrapper":
        return None
    try:
        from nn_models.torch_regressors import compute_input_gradient_importance

        return compute_input_gradient_importance(model, X_arr, max_samples=max_samples)
    except Exception:
        return None
