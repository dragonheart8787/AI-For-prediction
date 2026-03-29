#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從實驗紀錄、artifacts、批次訓練日誌讀取結果，輸出**適合大量實驗**的圖表與 CSV。

圖表類型：
  - 任務×模型 熱力圖（平均 RMSE）
  - 依模型 RMSE 箱形圖（多筆實驗分佈）
  - RMSE–R² 氣泡圖（氣泡大小＝樣本數、顏色＝模型）
  - 依任務分面：實驗序號 vs RMSE
  - 批次訓練每 job 耗時長條圖（若有 data/batch_train_log.jsonl）
  - 保留原有時間線、walk-forward、epoch、artifacts 長條等

使用：
  pip install -r requirements-viz.txt
  python scripts/plot_training_charts.py
  python scripts/plot_training_charts.py --runs-limit 0 --output-dir reports/charts

--runs-limit 0 表示讀取實驗 JSONL 內**全部** model_trained 紀錄（預設 0）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(
            "需要 matplotlib。請執行：pip install -r requirements-viz.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "PingFang TC",
        "Noto Sans CJK TC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_training_runs(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """limit<=0 表示全部紀錄。"""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("event_type") != "model_trained":
            continue
        if r.get("status") != "success":
            continue
        m = r.get("metrics") or {}
        out.append(
            {
                "ts": r.get("ts_iso") or r.get("timestamp") or "",
                "task": str(r.get("task_name", "")),
                "model": str(r.get("model_trained", r.get("model_requested", ""))),
                "rows": int(r.get("row_count", 0)),
                "features": int(r.get("feature_count", 0)),
                "train_rmse": _f(m.get("train_rmse")),
                "train_mae": _f(m.get("train_mae")),
                "train_r2": _f(m.get("train_r2")),
                "duration_ms": r.get("duration_ms"),
                "run_id": str(r.get("run_id", "")),
                "source": "experiment",
            }
        )
    if limit and limit > 0:
        return out[-limit:]
    return out


def parse_walk_forward(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("event") != "walk_forward":
            continue
        summ = r.get("summary") or {}
        if not summ:
            continue
        out.append(
            {
                "ts": r.get("ts_iso", ""),
                "task": r.get("task_id", ""),
                "model": r.get("model_requested", ""),
                "n_folds": summ.get("n_folds"),
                "mean_rmse": _f(summ.get("mean_rmse")),
                "mean_mae": _f(summ.get("mean_mae")),
                "mean_r2": _f(summ.get("mean_r2")),
            }
        )
    if limit and limit > 0:
        return out[-limit:]
    return out


def scan_artifact_metrics(artifacts_root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(os.path.abspath(artifacts_root), "**", "metrics.json")
    found: List[Dict[str, Any]] = []
    for p in glob.glob(pattern, recursive=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        parts = os.path.normpath(p).split(os.sep)
        task = parts[-3] if len(parts) >= 3 else ""
        run_id = parts[-2] if len(parts) >= 2 else ""
        model_guess = "?"
        cfgp = os.path.join(os.path.dirname(p), "config.json")
        if os.path.isfile(cfgp):
            try:
                with open(cfgp, "r", encoding="utf-8") as cf:
                    cj = json.load(cf)
                model_guess = str(cj.get("model", "?"))
            except (OSError, json.JSONDecodeError):
                pass
        found.append(
            {
                "path": p,
                "task": task,
                "run_id": run_id,
                "model": model_guess,
                "train_rmse": _f(m.get("train_rmse")),
                "train_mae": _f(m.get("train_mae")),
                "train_r2": _f(m.get("train_r2")),
            }
        )
    return found


def artifacts_as_runs(art: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """與 parse_training_runs 相同欄位，供合併畫圖。"""
    out: List[Dict[str, Any]] = []
    for a in art:
        if a.get("train_rmse") is None and a.get("train_r2") is None:
            continue
        out.append(
            {
                "ts": "",
                "task": a["task"],
                "model": a["model"],
                "rows": 0,
                "features": 0,
                "train_rmse": a["train_rmse"],
                "train_mae": a["train_mae"],
                "train_r2": a["train_r2"],
                "duration_ms": None,
                "run_id": a["run_id"],
                "source": "artifact",
            }
        )
    return out


def merge_runs_for_plot(
    exp_runs: List[Dict[str, Any]],
    art_runs: List[Dict[str, Any]],
    *,
    prefer: str = "both",
) -> List[Dict[str, Any]]:
    """prefer: experiment | artifact | both（兩者皆保留，圖表端可去重）。"""
    if prefer == "experiment":
        return list(exp_runs)
    if prefer == "artifact":
        return list(art_runs)
    # 去重：同 task+run_id 只保留 experiment
    seen = {(r["task"], r["run_id"]) for r in exp_runs if r.get("run_id")}
    merged = list(exp_runs)
    for r in art_runs:
        key = (r["task"], r["run_id"])
        if key in seen:
            continue
        merged.append(r)
    return merged


def parse_batch_train_log(path: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    return [r for r in rows if r.get("event_type") == "batch_train_job"]


def export_runs_csv(runs: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = [
        "source",
        "task",
        "model",
        "train_rmse",
        "train_mae",
        "train_r2",
        "rows",
        "features",
        "duration_ms",
        "run_id",
        "ts",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_heatmap_task_model(runs: List[Dict[str, Any]], out_path: str, plt: Any, metric: str = "train_rmse") -> bool:
    """任務×模型：格子內為該組合的平均 metric（越低越好用 viridis_r）。"""
    vals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in runs:
        k = (r["task"], r["model"])
        v = r.get(metric)
        if v is not None:
            vals[k].append(float(v))
    if len(vals) < 2:
        return False
    tasks = sorted({t for t, _ in vals.keys() if t})
    models = sorted({m for _, m in vals.keys() if m})
    if not tasks or not models:
        return False
    mat = np.full((len(tasks), len(models)), np.nan, dtype=float)
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            L = vals.get((t, m))
            if L:
                mat[i, j] = float(np.nanmean(L))
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), max(5, len(tasks) * 0.55)))
    cmap = "viridis_r" if metric == "train_rmse" else "viridis"
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([str(x)[:24] for x in tasks], fontsize=9)
    ax.set_title(f"任務 × 模型：平均 {metric}（格內數字＝多筆實驗平均）")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    # 註解數字
    if mat.size <= 120:
        for i in range(len(tasks)):
            for j in range(len(models)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center", color="w" if v < np.nanmedian(mat) else "k", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_boxplot_rmse_by_model(runs: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    by_m: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        if r["train_rmse"] is not None:
            by_m[str(r["model"])].append(float(r["train_rmse"]))
    data = []
    labels = []
    for m in sorted(by_m.keys()):
        if len(by_m[m]) >= 1:
            data.append(by_m[m])
            labels.append(f"{m}\n(n={len(by_m[m])})")
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.85)
    ax.set_ylabel("train RMSE 分佈")
    ax.set_title("各模型訓練 RMSE（箱形圖，適合大量重跑）")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_bubble_rmse_r2(runs: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    pts = [r for r in runs if r["train_rmse"] is not None and r["train_r2"] is not None]
    if len(pts) < 2:
        return False
    models = sorted({str(r["model"]) for r in pts})
    model_to_idx = {m: i for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in models:
        sub = [r for r in pts if str(r["model"]) == m]
        xs = [float(r["train_rmse"]) for r in sub]
        ys = [float(r["train_r2"]) for r in sub]
        ss = [max(30, min(400, int(r["rows"]) // 2 + 20)) if r["rows"] else 40 for r in sub]
        ci = model_to_idx[m] % 10
        ax.scatter(
            xs,
            ys,
            s=ss,
            alpha=0.65,
            label=m,
            color=f"C{ci}",
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_xlabel("train RMSE")
    ax.set_ylabel("train R²")
    ax.set_title("RMSE–R² 氣泡圖（氣泡大小 ~ 樣本數）")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_faceted_task_rmse(runs: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    """每個任務一個子圖：實驗序號 vs RMSE，多模型不同色。"""
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r["train_rmse"] is None or not r["task"]:
            continue
        by_task[r["task"]].append(r)
    if not by_task:
        return False
    tasks = sorted(by_task.keys())
    n = min(len(tasks), 12)
    tasks = tasks[:n]
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for idx, task in enumerate(tasks):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = sorted(by_task[task], key=lambda x: (x.get("ts") or "", x.get("run_id") or ""))
        models = sorted({str(x["model"]) for x in sub})
        for mi, m in enumerate(models):
            ys = [float(x["train_rmse"]) for x in sub if str(x["model"]) == m and x["train_rmse"] is not None]
            xs = list(range(len(ys)))
            if ys:
                ax.plot(xs, ys, "o-", label=m, color=f"C{mi % 10}", markersize=4, alpha=0.85)
        ax.set_title(str(task)[:28] + ("…" if len(str(task)) > 28 else ""), fontsize=9)
        ax.set_xlabel("該任務內第 n 筆")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
    for j in range(len(tasks), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)
    fig.suptitle("依任務：訓練 RMSE 走勢（大量實驗）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_batch_duration(batch_rows: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    if not batch_rows:
        return False
    labels = [
        f"{r.get('task', '')[:10]}/{r.get('model', '')}\n#{r.get('index')}"
        for r in batch_rows
    ]
    el = [float(r.get("elapsed_sec", 0) or 0) for r in batch_rows]
    ok = [r.get("exit_code", 1) == 0 for r in batch_rows]
    colors = ["#31a354" if o else "#cb181d" for o in ok]
    fig, ax = plt.subplots(figsize=(max(10, len(batch_rows) * 0.35), 5))
    ax.barh(range(len(batch_rows)), el, color=colors, edgecolor="white")
    ax.set_yticks(range(len(batch_rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("秒")
    ax.set_title("批次訓練：每 job 耗時（綠=成功 紅=失敗）")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_runs_timeline(runs: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    if not runs:
        return False
    rmse = [r["train_rmse"] for r in runs]
    if all(v is None for v in rmse):
        return False
    xs = list(range(len(runs)))
    max_ticks = 40
    step = max(1, len(runs) // max_ticks)
    labels = [f"{i}" for i in xs]
    fig, ax = plt.subplots(figsize=(min(24, 6 + len(runs) * 0.08), 4.5))
    ax.plot(xs, [v if v is not None else float("nan") for v in rmse], "-", color="#2c7fb8", lw=1, alpha=0.8)
    ax.scatter(
        xs,
        [v if v is not None else float("nan") for v in rmse],
        c="#08519c",
        s=12,
        alpha=0.7,
        zorder=3,
    )
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(labels[::step], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("train RMSE")
    ax.set_title(f"實驗序 RMSE（共 {len(runs)} 筆）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_r2_heatmap_task_model(runs: List[Dict[str, Any]], out_path: str, plt: Any) -> bool:
    return plot_heatmap_task_model(runs, out_path, plt, metric="train_r2")


def plot_epochs(by_run: Dict[str, List[Dict[str, Any]]], out_path: str, plt: Any, max_runs: int) -> bool:
    if not by_run:
        return False
    n = min(max_runs, len(by_run))
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), squeeze=False)
    for i, (rid, recs) in enumerate(list(by_run.items())[-n:]):
        ax = axes[i, 0]
        ep = [int(r.get("epoch", 0)) for r in recs]
        tl = [float(r.get("train_loss", 0)) for r in recs]
        ax.plot(ep, tl, label="train_loss", color="#3182bd")
        vl = [r.get("val_loss") for r in recs]
        if any(v is not None for v in vl):
            ax.plot(ep, [float(v) if v is not None else float("nan") for v in vl], label="val_loss", color="#e6550d")
        ax.set_title(f"Epoch run={rid[:14]}…")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def parse_epoch_logs(path: str, run_id: Optional[str], limit_runs: int) -> Dict[str, List[Dict[str, Any]]]:
    rows = _read_jsonl(path)
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("event_type") != "epoch":
            continue
        rid = str(r.get("run_id", ""))
        if run_id and rid != run_id:
            continue
        by_run.setdefault(rid, []).append(r)
    for rid in by_run:
        by_run[rid].sort(key=lambda x: int(x.get("epoch", 0)))
    if limit_runs > 0 and len(by_run) > limit_runs:
        keys = sorted(by_run.keys())[-limit_runs:]
        by_run = {k: by_run[k] for k in keys}
    return by_run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="訓練／實驗結果圖表（大量實驗版）")
    ap.add_argument("--output-dir", default="reports/charts", help="PNG / CSV 輸出目錄")
    ap.add_argument("--experiments", default="data/experiment_runs.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--epochs", default="data/epoch_logs.jsonl")
    ap.add_argument("--batch-log", default="data/batch_train_log.jsonl")
    ap.add_argument(
        "--runs-limit",
        type=int,
        default=0,
        help="實驗 JSONL 最多幾筆（0=全部）",
    )
    ap.add_argument(
        "--merge-sources",
        choices=["both", "experiment", "artifact"],
        default="both",
        help="合併實驗紀錄與 artifacts 的方式",
    )
    ap.add_argument("--epoch-run-id", default=None)
    ap.add_argument("--epoch-run-limit", type=int, default=6)
    ap.add_argument("--y-npy", default=None)
    ap.add_argument("--x-npy", default=None)
    ap.add_argument("--feature-idx", type=int, default=0)
    args = ap.parse_args()

    plt = _setup_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = _read_jsonl(args.experiments)
    exp_runs = parse_training_runs(rows, args.runs_limit)
    wf = parse_walk_forward(rows, args.runs_limit if args.runs_limit > 0 else 10_000)
    art = scan_artifact_metrics(args.artifacts)
    art_runs = artifacts_as_runs(art)
    runs = merge_runs_for_plot(exp_runs, art_runs, prefer=args.merge_sources)

    written: List[str] = []

    csv_path = os.path.join(args.output_dir, "training_runs_table.csv")
    export_runs_csv(runs, csv_path)
    written.append(csv_path)
    logger.info("已輸出 %s（%d 列）", csv_path, len(runs))

    def save(name: str, fn, *a, **kw) -> None:
        p = os.path.join(args.output_dir, name)
        if fn(*a, p, plt, **kw):
            written.append(p)
            logger.info("已輸出 %s", p)

    save("01_heatmap_task_x_model_rmse.png", plot_heatmap_task_model, runs)
    save("02_heatmap_task_x_model_r2.png", plot_r2_heatmap_task_model, runs)
    save("03_boxplot_rmse_by_model.png", plot_boxplot_rmse_by_model, runs)
    save("04_bubble_rmse_r2.png", plot_bubble_rmse_r2, runs)
    save("05_faceted_rmse_by_task.png", plot_faceted_task_rmse, runs)
    save("06_experiment_index_vs_rmse.png", plot_runs_timeline, runs)

    batch_path = args.batch_log
    if not os.path.isabs(batch_path):
        batch_path = os.path.join(REPO_ROOT, batch_path)
    batch_rows = parse_batch_train_log(batch_path)
    save("07_batch_job_duration.png", plot_batch_duration, batch_rows)

    # Walk-forward
    if wf:
        fig, ax = plt.subplots(figsize=(min(16, 4 + len(wf) * 0.35), 4.5))
        x = range(len(wf))
        ax.bar(x, [w["mean_rmse"] or 0 for w in wf], color="#fd8d3c", edgecolor="#d94801")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{w['task'][:8]}\n{w['model']}" for w in wf], fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("Walk-forward 平均 RMSE")
        ax.set_title("Walk-forward 紀錄")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.output_dir, "08_walk_forward_rmse.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
        logger.info("已輸出 %s", p)

    ep = parse_epoch_logs(args.epochs, args.epoch_run_id, args.epoch_run_limit)
    p = os.path.join(args.output_dir, "09_epoch_loss_curves.png")
    if plot_epochs(ep, p, plt, args.epoch_run_limit):
        written.append(p)
        logger.info("已輸出 %s", p)

    if art:
        tail = art[-40:]
        fig, ax = plt.subplots(figsize=(12, max(4, len(tail) * 0.15)))
        ax.barh(
            [f"{a['task'][:12]}/{a['run_id'][:6]}" for a in tail],
            [a["train_rmse"] or 0 for a in tail],
            color="#9e9ac8",
        )
        ax.set_xlabel("train RMSE（metrics.json）")
        ax.set_title("Artifacts 最近筆數")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.output_dir, "10_artifacts_rmse_recent.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
        logger.info("已輸出 %s", p)

    if args.y_npy:
        import numpy as np

        y = np.load(args.y_npy).ravel()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(y, bins=min(60, max(12, int(np.sqrt(len(y))))), color="#74c476", edgecolor="white")
        ax.set_title("目標 y 分佈")
        fig.tight_layout()
        p = os.path.join(args.output_dir, "11_data_y_distribution.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
        logger.info("已輸出 %s", p)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_experiment_runs": len(exp_runs),
                "n_artifact_runs": len(art_runs),
                "n_merged_for_charts": len(runs),
                "n_batch_jobs": len(batch_rows),
                "outputs": written,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("摘要 %s", summary_path)

    pngs = [w for w in written if w.endswith(".png")]
    if not pngs:
        print(
            "尚未產生 PNG：請先執行訓練或批次訓練，或檢查 --experiments / --artifacts 路徑。",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
