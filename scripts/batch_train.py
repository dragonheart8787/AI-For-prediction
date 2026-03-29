#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大量批次訓練：依 YAML 或 --matrix 連續執行 crawler_train_pipeline.py。

每筆 job 會呼叫一次子程序，實驗紀錄仍寫入 data/experiment_runs.jsonl（除非 job 設 no_experiment_log）。
執行摘要寫入 data/batch_train_log.jsonl。

範例：
  python scripts/batch_train.py --config config/batch_train.example.yaml --dry-run
  python scripts/batch_train.py --matrix-tasks stock_price_next,temperature_next --matrix-models linear,xgboost
  python scripts/batch_train.py --all-tasks --models linear,lightgbm --continue-on-fail
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_existing_path(path: str) -> str:
    """依目前工作目錄或專案根目錄尋找設定檔。"""
    if os.path.isfile(path):
        return os.path.abspath(path)
    alt = os.path.join(REPO_ROOT, path)
    if os.path.isfile(alt):
        return alt
    return os.path.abspath(path)


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("批次設定需要 PyYAML：pip install pyyaml") from e
    resolved = _resolve_existing_path(path)
    if not os.path.isfile(resolved):
        print(
            f"找不到設定檔：{path}\n"
            f"已嘗試：{os.path.abspath(path)} 與 {os.path.join(REPO_ROOT, path)}\n\n"
            "請擇一：\n"
            "  1) 使用倉庫內預設檔：python scripts/batch_train.py --config config/batch_train.yaml\n"
            "  2) 複製範例後再跑：copy config\\batch_train.example.yaml config\\batch_train.yaml\n"
            "  3) 直接指定範例：--config config/batch_train.example.yaml",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover_tasks(schema_path: str) -> List[str]:
    cfg = _load_yaml(schema_path)
    return list((cfg.get("prediction_tasks") or {}).keys())


def _expand_matrix(
    matrix: Dict[str, Any],
    exclude: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    tasks = list(matrix.get("tasks") or [])
    models = list(matrix.get("models") or [])
    ex = {(e.get("task"), e.get("model")) for e in exclude}
    jobs: List[Dict[str, Any]] = []
    for t in tasks:
        for m in models:
            if (t, m) in ex:
                continue
            jobs.append({"task": t, "model": m})
    return jobs


def _merge_job(defaults: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    out.update(job)
    return out


def job_to_command(
    merged: Dict[str, Any],
    schema: str,
    python_exe: str,
) -> List[str]:
    task = merged["task"]
    model = merged["model"]
    cmd: List[str] = [
        python_exe,
        os.path.join(REPO_ROOT, "crawler_train_pipeline.py"),
        task,
        "--model",
        str(model),
        "--schema",
        os.path.join(REPO_ROOT, schema) if not os.path.isabs(schema) else schema,
    ]

    def bflag(key: str, arg: str) -> None:
        if merged.get(key):
            cmd.append(arg)

    bflag("rich_features", "--rich-features")
    bflag("walk_forward", "--walk-forward")
    bflag("no_normalize", "--no-normalize")
    bflag("no_experiment_log", "--no-experiment-log")
    bflag("replay", "--replay")
    bflag("wf_allow_deep", "--wf-allow-deep")
    bflag("automl_include_deep", "--automl-include-deep")
    bflag("train_all", "--train-all")

    if merged.get("preset") and str(merged["preset"]) != "default":
        cmd.extend(["--preset", str(merged["preset"])])
    if merged.get("period"):
        cmd.extend(["--period", str(merged["period"])])
    if merged.get("symbol"):
        cmd.extend(["--symbol", str(merged["symbol"])])
    if merged.get("models_dir"):
        cmd.extend(["--models-dir", str(merged["models_dir"])])
    if merged.get("data_dir"):
        cmd.extend(["--data-dir", str(merged["data_dir"])])
    if merged.get("memory_path"):
        cmd.extend(["--memory-path", str(merged["memory_path"])])
    if merged.get("automl_trials") is not None:
        cmd.extend(["--automl-trials", str(int(merged["automl_trials"]))])
    if merged.get("automl_timeout") is not None:
        cmd.extend(["--automl-timeout", str(float(merged["automl_timeout"]))])
    if merged.get("wf_splits") is not None:
        cmd.extend(["--wf-splits", str(int(merged["wf_splits"]))])

    extra = merged.get("extra_args")
    if isinstance(extra, list):
        cmd.extend(str(x) for x in extra)
    return cmd


def _append_batch_log(rec: Dict[str, Any], path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    rec["ts_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def collect_jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], str]:
    schema = args.schema

    if args.config:
        cfg = _load_yaml(args.config)
        schema = cfg.get("schema", schema)
        defaults = dict(cfg.get("defaults") or {})
        jobs = list(cfg.get("jobs") or [])
        matrix = cfg.get("matrix") or {}
        exclude = list(cfg.get("exclude") or [])
        if not jobs and matrix.get("tasks") and matrix.get("models"):
            jobs = _expand_matrix(matrix, exclude)
        merged = [_merge_job(defaults, j) for j in jobs]
        return merged, schema

    jobs: List[Dict[str, Any]] = []

    if args.all_tasks:
        tasks = _discover_tasks(os.path.join(REPO_ROOT, schema) if not os.path.isabs(schema) else schema)
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for t in tasks:
            for m in models:
                jobs.append({"task": t, "model": m})
    elif args.matrix_tasks and args.matrix_models:
        tasks = [x.strip() for x in args.matrix_tasks.split(",") if x.strip()]
        models = [x.strip() for x in args.matrix_models.split(",") if x.strip()]
        jobs = _expand_matrix({"tasks": tasks, "models": models}, [])
    else:
        raise SystemExit("請指定 --config，或 --matrix-tasks 與 --matrix-models，或 --all-tasks --models ...")

    defaults: Dict[str, Any] = {}
    if args.rich_features:
        defaults["rich_features"] = True
    if args.walk_forward:
        defaults["walk_forward"] = True
    if args.preset and args.preset != "default":
        defaults["preset"] = args.preset
    if args.period:
        defaults["period"] = args.period

    merged = [_merge_job(defaults, j) for j in jobs]
    return merged, schema


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict-AI 批次訓練")
    ap.add_argument("--config", type=str, default=None, help="YAML 設定檔")
    ap.add_argument("--schema", type=str, default="config/prediction_schema.yaml")
    ap.add_argument("--matrix-tasks", type=str, default=None, help="逗號分隔 task id")
    ap.add_argument("--matrix-models", type=str, default=None, help="逗號分隔模型名")
    ap.add_argument("--all-tasks", action="store_true", help="對 schema 內全部任務訓練")
    ap.add_argument("--models", type=str, default="linear", help="與 --all-tasks 併用，逗號分隔")
    ap.add_argument("--rich-features", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--preset", type=str, default="default", choices=["default", "strong"])
    ap.add_argument("--period", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只列出將執行的指令")
    ap.add_argument("--continue-on-fail", action="store_true", help="單筆失敗仍繼續")
    ap.add_argument("--log-path", type=str, default="data/batch_train_log.jsonl")
    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    if not args.config and not args.all_tasks and not (args.matrix_tasks and args.matrix_models):
        ap.print_help()
        print(
            "\n範例：python scripts/batch_train.py --matrix-tasks stock_price_next --matrix-models linear,xgboost",
            file=sys.stderr,
        )
        raise SystemExit(2)

    jobs, schema = collect_jobs(args)
    if not jobs:
        print("沒有任何 job。", file=sys.stderr)
        raise SystemExit(1)

    print(f"共 {len(jobs)} 筆訓練任務（schema={schema}）\n")

    ok, fail = 0, 0
    for i, merged in enumerate(jobs, 1):
        cmd = job_to_command(merged, schema, args.python)
        print(f"[{i}/{len(jobs)}] {' '.join(cmd[2:5])} ...")  # task model 片段
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue

        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=False,
            )
            elapsed = time.perf_counter() - t0
            rec = {
                "event_type": "batch_train_job",
                "index": i,
                "total": len(jobs),
                "task": merged.get("task"),
                "model": merged.get("model"),
                "exit_code": r.returncode,
                "elapsed_sec": round(elapsed, 3),
                "cmd": cmd,
            }
            _append_batch_log(rec, os.path.join(REPO_ROOT, args.log_path))
            if r.returncode == 0:
                ok += 1
            else:
                fail += 1
                print(f"  !! exit {r.returncode} ({elapsed:.1f}s)")
                if not args.continue_on_fail:
                    print("已中止（加 --continue-on-fail 可略過錯誤繼續）", file=sys.stderr)
                    raise SystemExit(r.returncode)
        except Exception as e:
            fail += 1
            _append_batch_log(
                {
                    "event_type": "batch_train_job",
                    "index": i,
                    "task": merged.get("task"),
                    "model": merged.get("model"),
                    "exit_code": -1,
                    "error": str(e),
                },
                os.path.join(REPO_ROOT, args.log_path),
            )
            if not args.continue_on_fail:
                raise

    if args.dry_run:
        print("\n（dry-run 未實際執行）")
        return

    print(f"\n完成：成功 {ok}，失敗 {fail}。摘要：{args.log_path}")


if __name__ == "__main__":
    main()
