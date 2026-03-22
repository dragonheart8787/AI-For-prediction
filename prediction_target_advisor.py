#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預測目標顧問：當使用者提出特定預測目標時，AI 知道需要找什麼資料

功能：
- 輸入：使用者的預測目標描述（如「我想預測股價」「台北氣溫」）
- 輸出：對應的任務、需爬取的資料來源、特徵欄位、建議參數
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_keywords(path: str = "config/prediction_target_keywords.yaml") -> Dict[str, List[str]]:
    """載入關鍵字對應"""
    if yaml is None:
        raise ImportError("請安裝 pyyaml: pip install pyyaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("target_keywords", {})


def load_schema(path: str = "config/prediction_schema.yaml") -> Dict[str, Any]:
    """載入預測任務 schema"""
    if yaml is None:
        raise ImportError("請安裝 pyyaml: pip install pyyaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def match_prediction_target(
    user_input: str,
    keywords: Dict[str, List[str]],
) -> Optional[str]:
    """
    根據使用者輸入匹配預測任務。
    回傳最匹配的 task_id，若無則回傳 None。
    """
    if not user_input or not keywords:
        return None

    text = user_input.strip().lower()
    # 移除常見前綴
    for prefix in ["我想", "我要", "預測", "預估", "估計", "分析"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    best_task: Optional[str] = None
    best_len = 0

    for task_id, kws in keywords.items():
        for kw in kws:
            kw_lower = kw.lower()
            # 僅當「關鍵字出現在使用者輸入」時匹配（避免「股價」誤匹配「股價加新聞」）
            if kw_lower in text:
                if len(kw) > best_len:
                    best_len = len(kw)
                    best_task = task_id

    return best_task


def get_data_requirements_for_task(
    task_id: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """取得指定任務的資料需求"""
    tasks = schema.get("prediction_tasks", {})
    if task_id not in tasks:
        return {"task_id": task_id, "found": False}

    task = tasks[task_id]
    sources = []

    for ds in task.get("data_sources", []):
        conn = ds.get("connector", "")
        features = ds.get("features", [])
        params = ds.get("params", {})

        # 資料來源中文說明
        conn_desc = {
            "yahoo": "Yahoo Finance 股價資料",
            "eia": "EIA 能源資料",
            "newsapi": "NewsAPI 新聞情感",
            "open_meteo": "Open-Meteo 天氣資料",
            "owid": "OWID 疫情資料",
        }.get(conn, conn)

        sources.append({
            "connector": conn,
            "description": conn_desc,
            "features": features,
            "params": params,
        })

    return {
        "task_id": task_id,
        "found": True,
        "display_name": task.get("display_name", task_id),
        "domain": task.get("domain", "custom"),
        "target_source": task.get("target_source", ""),
        "target_horizon": task.get("target_horizon", 1),
        "data_sources": sources,
    }


def advise_for_prediction_target(
    user_input: str,
    schema_path: str = "config/prediction_schema.yaml",
    keywords_path: str = "config/prediction_target_keywords.yaml",
) -> Dict[str, Any]:
    """
    根據使用者提出的預測目標，回傳需要找什麼資料。

    Returns:
        {
            "matched": bool,
            "task_id": str,
            "display_name": str,
            "data_sources": [...],
            "summary": str,  # 人類可讀的建議摘要
        }
    """
    keywords = load_keywords(keywords_path)
    schema = load_schema(schema_path)

    # 預設任務
    tasks = schema.get("prediction_tasks", {})
    default_task = "stock_price_next"
    if os.path.exists(keywords_path):
        with open(keywords_path, "r", encoding="utf-8") as f:
            kw_data = yaml.safe_load(f) or {}
        default_task = kw_data.get("default_task", default_task)

    task_id = match_prediction_target(user_input, keywords)
    if task_id is None:
        task_id = default_task
        matched = False
    else:
        matched = True

    req = get_data_requirements_for_task(task_id, schema)

    if not req.get("found"):
        return {
            "matched": False,
            "task_id": task_id,
            "display_name": "未知任務",
            "data_sources": [],
            "summary": f"無法找到對應任務，請確認預測目標。可用任務: {list(schema.get('prediction_tasks', {}).keys())}",
        }

    # 建立人類可讀摘要
    parts = [f"要預測「{req['display_name']}」，需要爬取以下資料："]
    for ds in req["data_sources"]:
        feats = "、".join(ds["features"])
        parts.append(f"  - {ds['description']}：{feats}")
    summary = "\n".join(parts)

    return {
        "matched": matched,
        "task_id": task_id,
        "display_name": req["display_name"],
        "domain": req["domain"],
        "target_source": req["target_source"],
        "data_sources": req["data_sources"],
        "summary": summary,
        "requirements": req,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="預測目標顧問：輸入預測目標，取得建議的資料來源",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="預測目標描述（如：我想預測股價、台北氣溫）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="輸出 JSON 格式",
    )
    args = parser.parse_args()

    if not args.target:
        print("請輸入預測目標，例如：")
        print("  python prediction_target_advisor.py \"我想預測股價\"")
        print("  python prediction_target_advisor.py \"台北氣溫\"")
        print("  python prediction_target_advisor.py \"能源需求\"")
        return

    result = advise_for_prediction_target(args.target)

    if args.json:
        import json
        # 移除 requirements 中的冗長內容以簡化輸出
        out = {k: v for k, v in result.items() if k != "requirements"}
        out["data_sources"] = [
            {"connector": ds["connector"], "description": ds["description"], "features": ds["features"]}
            for ds in result.get("data_sources", [])
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=== 預測目標顧問 ===\n")
        print(f"輸入: {args.target}")
        print(f"匹配任務: {result['task_id']} ({result['display_name']})")
        print(f"匹配成功: {'是' if result['matched'] else '否（使用預設）'}\n")
        print(result["summary"])
        print("\n執行訓練指令:")
        print(f"  python crawler_train_pipeline.py {result['task_id']}")


if __name__ == "__main__":
    main()
