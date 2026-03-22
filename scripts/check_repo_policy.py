#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
倉庫政策檢查（供 CI / 本機 pre-push）：
- 禁止在 git 索引根層出現已歸檔的目錄名（避免根目錄再次變髒）
- VERSION、pyproject.toml、CHANGELOG 首條、README 版號敘述需一致
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 這些名稱若出現在「已追蹤檔案的第一層路徑」→ 視為根目錄污染
FORBIDDEN_ROOT_NAMES = frozenset(
    {
        "archive",
        "accelerators",
        "agi_images",
        "agi_storage",
        "collected_data",
        "configs",
        "cpu_optimization",
        "data_pipeline",
        "demo_export",
        "enhanced_data",
        "enhanced_models",
        "guides",
        "knowledge_distillation",
        "paper",
        "pretrained_models",
        "reinforcement_learning",
        "serving",
        "super_fusion_storage",
        "super_enhanced_ts_results",
        "xai",
    }
)


def _git_ls_files() -> list[str]:
    r = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _top_level_roots(paths: list[str]) -> set[str]:
    roots: set[str] = set()
    for p in paths:
        first = p.split("/", 1)[0]
        if first:
            roots.add(first)
    return roots


def check_root_clean(tracked: list[str]) -> None:
    roots = _top_level_roots(tracked)
    bad = sorted(FORBIDDEN_ROOT_NAMES & roots)
    if bad:
        print(
            "ERROR: 根層不應追蹤下列目錄（歷史內容請放分支 legacy-archive）:",
            ", ".join(bad),
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("OK: 根層無禁止目錄")


def read_version() -> str:
    v = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not re.match(r"^\d+\.\d+\.\d+", v):
        print("ERROR: VERSION 格式應為 semver 如 0.9.2", file=sys.stderr)
        raise SystemExit(1)
    return v


def check_pyproject(version: str) -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        print("ERROR: pyproject.toml 找不到 version =", file=sys.stderr)
        raise SystemExit(1)
    if m.group(1) != version:
        print(
            f"ERROR: pyproject.toml version={m.group(1)!r} 與 VERSION={version!r} 不一致",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("OK: pyproject.toml version 與 VERSION 一致")


def check_changelog(version: str) -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r"^##\s+v?(\d+\.\d+\.\d+)", text, re.MULTILINE)
    if not m:
        print("ERROR: CHANGELOG.md 找不到 ## vX.Y.Z 首條", file=sys.stderr)
        raise SystemExit(1)
    if m.group(1) != version:
        print(
            f"ERROR: CHANGELOG 首條版本 {m.group(1)!r} 與 VERSION={version!r} 不一致",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("OK: CHANGELOG 首條版本與 VERSION 一致")


def check_readme(version: str) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    if f"**`{version}`**" not in text:
        print("ERROR: README.md 應含 **`{ver}`** 形式版號（與 VERSION 一致）".replace("{ver}", version), file=sys.stderr)
        raise SystemExit(1)
    print("OK: README 含版號字樣")


def main() -> None:
    tracked = _git_ls_files()
    check_root_clean(tracked)
    version = read_version()
    check_pyproject(version)
    check_changelog(version)
    check_readme(version)
    print(f"repo_policy_ok version={version}")


if __name__ == "__main__":
    main()
