import logging
from dataclasses import dataclass, field

import numpy as np
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


def _normalize_datetime_index_for_merge(idx: Any) -> Any:
    """
    讓 merge_asof 的左右索引 dtype 一致：pandas 不允許 naive 與 tz-aware 混用。
    tz-aware 一律轉成 UTC 再去掉時區（保留同一瞬間的牆鐘時間，以 naive 表示）。
    """
    if pd is None:
        return idx
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


@dataclass
class AlignSummary:
    """多源對齊摘要（可追溯資料品質）。"""

    rows_before: Dict[str, int] = field(default_factory=dict)
    rows_after: int = 0
    nan_ratio: float = 0.0
    source_match_ratio: Dict[str, float] = field(default_factory=dict)


def infer_timestamp_key(rows: Iterable[Dict[str, Any]]) -> str:
    candidates = [
        "timestamp", "time", "date", "datetime", "ts"
    ]
    rows_list: List[Dict[str, Any]] = list(rows)
    if not rows_list:
        return "timestamp"
    keys = set().union(*[r.keys() for r in rows_list])
    for k in candidates:
        if k in keys:
            return k
    return "timestamp"


def rows_to_features(rows: List[Dict[str, Any]], timestamp_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k != timestamp_key}
        rec["timestamp"] = r.get(timestamp_key)
        out.append(rec)
    return out


def align_source_frames(
    source_frames: Dict[str, Any],
    target_source: str,
    *,
    timestamp_col: str = "timestamp",
    freq: str = "1D",
    join_strategy: str = "left",
    tolerance: str = "36h",
    missing_policy: str = "ffill",
    prevent_leakage: bool = True,
) -> Tuple[Any, AlignSummary]:
    """
    以 DataFrame 為輸入的多源對齊（顯式策略）。

    - ``prevent_leakage=True``：副表僅 ``merge_asof(..., direction="backward")``，
      且**禁止** ``bfill``（避免未來資訊回填）。
    """
    summary = AlignSummary()
    if pd is None:
        raise RuntimeError("align_source_frames 需要 pandas")

    if not source_frames:
        return pd.DataFrame(), summary

    for name, df in source_frames.items():
        summary.rows_before[name] = int(len(df)) if hasattr(df, "__len__") else 0

    if target_source not in source_frames:
        target_source = next(iter(source_frames))

    main_df = source_frames[target_source].copy()
    main_df[timestamp_col] = pd.to_datetime(main_df[timestamp_col], errors="coerce")
    main_df = main_df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    main_df = main_df.set_index(timestamp_col)
    main_df = main_df[~main_df.index.duplicated(keep="first")]
    main_df.index = _normalize_datetime_index_for_merge(main_df.index)
    main_df = main_df.resample(freq).asfreq()
    main_df.index.name = timestamp_col

    pol = missing_policy
    if prevent_leakage and pol == "bfill":
        logger.warning("prevent_leakage=True：已將 missing_policy 由 bfill 改為 ffill")
        pol = "ffill"

    exclude = {timestamp_col}
    aux_columns_added: List[str] = []
    per_source_hits: Dict[str, List[float]] = {}

    for source_name, aux_raw in source_frames.items():
        if source_name == target_source:
            continue
        aux_df = aux_raw.copy()
        aux_df[timestamp_col] = pd.to_datetime(aux_df[timestamp_col], errors="coerce")
        aux_df = aux_df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        aux_df = aux_df.set_index(timestamp_col)
        aux_df = aux_df[~aux_df.index.duplicated(keep="first")]
        aux_df.index = _normalize_datetime_index_for_merge(aux_df.index)
        feats = [c for c in aux_df.columns if c not in exclude]
        if not feats:
            continue

        merged = pd.merge_asof(
            main_df.sort_index(),
            aux_df[feats].sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(tolerance) if tolerance else None,
        )
        hits: List[float] = []
        for c in feats:
            col_name = f"{source_name}__{c}" if c in main_df.columns else c
            series = merged[c]
            main_df[col_name] = series
            aux_columns_added.append(col_name)
            hits.append(float(series.notna().mean()) if len(series) else 0.0)
        if hits:
            per_source_hits[source_name] = hits

    if join_strategy == "inner" and aux_columns_added:
        main_df = main_df.dropna(subset=aux_columns_added)

    if pol == "ffill":
        main_df = main_df.ffill()
    elif pol == "bfill" and not prevent_leakage:
        main_df = main_df.bfill()
    elif pol == "mean":
        main_df = main_df.fillna(main_df.mean(numeric_only=True))

    summary.rows_after = int(len(main_df))
    num = main_df.select_dtypes(include=["number"])
    if num.size:
        arr = np.asarray(num.values, dtype=float)
        summary.nan_ratio = float(np.mean(np.isnan(arr)))
    else:
        summary.nan_ratio = 0.0

    main_df = main_df.fillna(0)

    for sn, ratios in per_source_hits.items():
        summary.source_match_ratio[sn] = float(sum(ratios) / max(len(ratios), 1))

    main_df = main_df.reset_index()
    main_df[timestamp_col] = main_df[timestamp_col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return main_df, summary


def align_sources(
    rows_by_source: Dict[str, List[Dict[str, Any]]],
    target_source: str,
    freq: str = "1D",
    join_strategy: Literal["left", "inner"] = "left",
    join_tolerance: str = "36h",
    missing_policy: Literal["ffill", "bfill", "zero", "mean"] = "ffill",
    timestamp_key: str = "timestamp",
    prevent_leakage: bool = True,
    log_summary: bool = False,
    return_summary: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], AlignSummary]]:
    """
    將多資料源依時間窗對齊：以 target_source 為主表，其他 source 做 backward merge_asof。

    ``prevent_leakage=True`` 時禁用 bfill，避免未來欄位倒灌。
    """
    if pd is None:
        all_rows: List[Dict[str, Any]] = []
        for rows in rows_by_source.values():
            all_rows.extend(rows)
        if all_rows and timestamp_key in all_rows[0]:
            all_rows.sort(key=lambda r: str(r.get(timestamp_key, "")))
        return all_rows

    if not rows_by_source:
        return [] if not return_summary else ([], AlignSummary())

    dfs = {k: pd.DataFrame(v) for k, v in rows_by_source.items()}
    main_df, summary = align_source_frames(
        dfs,
        target_source,
        timestamp_col=timestamp_key,
        freq=freq,
        join_strategy=join_strategy,
        tolerance=join_tolerance,
        missing_policy=missing_policy,
        prevent_leakage=prevent_leakage,
    )
    out = main_df.to_dict(orient="records")
    if log_summary:
        logger.info(
            "align_sources summary: before=%s after=%d nan_ratio=%.4f match=%s",
            summary.rows_before,
            summary.rows_after,
            summary.nan_ratio,
            summary.source_match_ratio,
        )
    if return_summary:
        return out, summary
    return out


