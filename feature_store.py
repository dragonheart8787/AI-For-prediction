from typing import Any, Dict, Iterable, List
import json
import os


class JSONLFeatureStore:
    """極簡 Feature Store：以 JSONL 追加寫入方式儲存 rows。"""

    def __init__(self, path: str = "data_store.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

    def load_all(self, limit: int = 100000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out


