from typing import Any, Dict, Iterable, List, Union

from .base import DataConnector


class RESTConnector(DataConnector):
    """通用 REST 連接器：GET JSON，並以路徑映射擷取欄位。

    參數：
      - url: 來源 URL（GET）
      - query: dict 轉為 query string
      - root_path: JSON 內資料陣列的路徑（例如 "data.items"）
      - fields: 欲擷取欄位的路徑映射，例：{"timestamp":"time", "price":"close"}
    回傳：list[dict]
    """

    def __init__(self, timeout: int = 15, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="rest", timeout=timeout, max_retries=max_retries, **kwargs)

    def _get_by_path(self, obj: Any, path: str) -> Any:
        cur = obj
        for seg in path.split('.') if path else []:
            if isinstance(cur, dict):
                cur = cur.get(seg)
            elif isinstance(cur, list):
                try:
                    idx = int(seg)
                    cur = cur[idx]
                except Exception:
                    return None
            else:
                return None
        return cur

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        import urllib.parse
        url = str(kwargs["url"]).strip()
        query = kwargs.get("query") or {}
        root_path = kwargs.get("root_path") or ""
        fields: Dict[str, str] = kwargs.get("fields") or {}

        data = self._request_json(url, params=query if query else None, task=url[:50])

        root = self._get_by_path(data, root_path) if root_path else data
        if not isinstance(root, list):
            root = [root]

        rows: List[Dict[str, Any]] = []
        for item in root:
            rec: Dict[str, Any] = {}
            for out_key, src_path in fields.items():
                rec[out_key] = self._get_by_path(item, src_path)
            rows.append(rec)
        return rows


