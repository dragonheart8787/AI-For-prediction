"""資料連接器基底：retry、timeout、backoff 共用層。"""
import json
import logging
import random
import socket
import time
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ConnectorError(Exception):
    """連接器基底錯誤。"""


class ConnectorTimeoutError(ConnectorError):
    """請求逾時。"""


class ConnectorRateLimitError(ConnectorError):
    """HTTP 429 等限流。"""


class ConnectorSchemaError(ConnectorError):
    """回傳結構不符合預期。"""


class ConnectorAuthError(ConnectorError):
    """401/403 等認證失敗。"""


class ConnectorUnavailableError(ConnectorError):
    """遠端不可用（5xx 等）。"""


class BaseConnector:
    """具 retry、timeout、backoff 的連接器基底。各 connector 只做 mapping，transport 由此提供。"""

    def __init__(
        self,
        name: str = "connector",
        timeout: int = 15,
        max_retries: int = 3,
        base_delay: float = 0.8,
        jitter: bool = True,
    ) -> None:
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.jitter = jitter

    def _retry(self, fn: Callable[[], Any], task: str = "") -> Any:
        """執行 fn，失敗時指數退避重試。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if isinstance(e, (ConnectorAuthError, ConnectorSchemaError)):
                    raise
                last_err = e
                if attempt >= self.max_retries:
                    logger.warning(
                        "connector=%s task=%s attempt=%d failed: %s",
                        self.name, task, attempt + 1, e,
                    )
                    raise RuntimeError(f"connector {self.name} request failed: {last_err}") from last_err
                delay = self.base_delay * (2 ** attempt)
                if self.jitter:
                    delay += random.uniform(0, 0.3)
                logger.debug("connector=%s retry in %.2fs: %s", self.name, delay, e)
                time.sleep(delay)
        raise RuntimeError(f"connector {self.name} request failed: {last_err}") from last_err

    def _request_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        task: str = "",
    ) -> Any:
        """HTTP GET JSON，支援 retry on 429/5xx。"""
        import urllib.parse
        import urllib.request
        from urllib.error import HTTPError, URLError
        params = params or {}
        headers = headers or {}

        def _do() -> Any:
            full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
            req = urllib.request.Request(full_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as e:
                if e.code == 429:
                    raise ConnectorRateLimitError(f"HTTP 429 {url[:80]}") from e
                if e.code in (401, 403):
                    raise ConnectorAuthError(f"HTTP {e.code} {url[:80]}") from e
                if e.code in RETRYABLE_STATUS:
                    logger.warning("connector=%s url=%s status=%d", self.name, url[:80], e.code)
                    raise ConnectorUnavailableError(f"HTTP {e.code}") from e
                raise
            except socket.timeout as e:
                raise ConnectorTimeoutError(str(e)) from e
            except URLError as e:
                if "timed out" in str(e).lower():
                    raise ConnectorTimeoutError(str(e)) from e
                raise
        try:
            return self._retry(_do, task=task)
        except ConnectorError:
            raise
        except RuntimeError as e:
            if "retryable" in str(e).lower():
                raise ConnectorUnavailableError(str(e)) from e
            raise

class DataConnector(BaseConnector):
    """資料連接器介面：輸出可被 UnifiedPredictor 接受的 list[dict] 結構。"""

    def __init__(self, name: str = "connector", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
