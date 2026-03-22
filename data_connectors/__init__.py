from .base import DataConnector
from .open_meteo import OpenMeteoConnector
from .owid import OWIDConnector
from .rest_generic import RESTConnector

try:
    from .yahoo import YahooFinanceConnector
except ImportError:
    YahooFinanceConnector = None  # type: ignore
try:
    from .eia import EIAConnector
except ImportError:
    EIAConnector = None  # type: ignore
try:
    from .newsapi import NewsAPIConnector
except ImportError:
    NewsAPIConnector = None  # type: ignore

__all__ = [
    "DataConnector",
    "OpenMeteoConnector",
    "OWIDConnector",
    "RESTConnector",
    "YahooFinanceConnector",
    "EIAConnector",
    "NewsAPIConnector",
]


