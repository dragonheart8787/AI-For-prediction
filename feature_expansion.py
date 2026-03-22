#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
時序表格特徵擴展：滯後、滾動均值／標準差。
用於 OHLCV 等數值欄，提升樹模型可表達的非線性與慣性。

需 pandas；未安裝時函式會原樣返回 X。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EXPAND_COLS: Set[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "temperature_2m",
        "relative_humidity_2m",
        "new_cases",
        "new_deaths",
        "sentiment",
    }
)


def expand_timeseries_features(
    X: np.ndarray,
    feature_names: List[str],
    *,
    lag_steps: Sequence[int] = (1, 2, 3, 5),
    rolling_windows: Sequence[int] = (3, 5),
    expand_columns: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    對指定欄位增加 lag_*、rollmean_*、rollstd_* 欄位（列與原 X 對齊）。
    NaN 以 0 填補（序列開頭滯後）。
    """
    if X.size == 0 or not feature_names:
        return X, list(feature_names)

    try:
        import pandas as pd
    except ImportError:
        logger.warning("未安裝 pandas，略過 feature_expansion")
        return X, list(feature_names)

    cols = expand_columns if expand_columns is not None else DEFAULT_EXPAND_COLS
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=feature_names)
    extra_names: List[str] = []

    for col in feature_names:
        if col not in cols or col not in df.columns:
            continue
        s = df[col]
        for L in lag_steps:
            if L <= 0:
                continue
            name = f"{col}_lag{L}"
            df[name] = s.shift(L)
            extra_names.append(name)
        for w in rolling_windows:
            if w < 2:
                continue
            mname = f"{col}_rollmean{w}"
            sname = f"{col}_rollstd{w}"
            df[mname] = s.rolling(window=w, min_periods=1).mean()
            df[sname] = s.rolling(window=w, min_periods=1).std()
            extra_names.extend([mname, sname])

    df = df.fillna(0.0)
    all_names = list(feature_names) + [n for n in extra_names if n in df.columns]
    # 欄位順序：原始 + 依建立順序的衍生欄
    ordered = list(feature_names) + [n for n in df.columns if n not in feature_names]
    X_out = df[ordered].to_numpy(dtype=np.float64)
    return X_out, ordered


def merge_expansion_config(
    task_cfg: Optional[Dict[str, Any]],
    cli_rich: bool,
) -> Optional[Dict[str, Any]]:
    """合併 YAML feature_expansion 與 CLI --rich-features。"""
    base: Dict[str, Any] = dict(task_cfg or {})
    if cli_rich:
        base["enabled"] = True
    if not base.get("enabled"):
        return None
    ec = base.get("expand_columns")
    if ec is not None:
        ec = frozenset(ec)
    return {
        "enabled": True,
        "lag_steps": tuple(base.get("lag_steps", [1, 2, 3, 5])),
        "rolling_windows": tuple(base.get("rolling_windows", [3, 5])),
        "expand_columns": ec,
    }
