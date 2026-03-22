"""
Data Connectors 測試套件
測試各資料連接器的 fetch API、離線降級與 retry 行為
"""
import pytest
from unittest.mock import patch, MagicMock
from data_connectors import (
    YahooFinanceConnector,
    OpenMeteoConnector,
    EIAConnector,
    OWIDConnector,
    NewsAPIConnector,
    RESTConnector,
)


class TestDataConnectors:
    """資料連接器測試"""

    def test_yahoo_fetch_returns_list(self):
        """Yahoo 回傳 list[dict]，含 timestamp 與價量欄位"""
        connector = YahooFinanceConnector()
        result = list(connector.fetch(symbol="AAPL", period="1mo"))
        assert isinstance(result, list)
        if result:
            r = result[0]
            assert "timestamp" in r
            assert "close" in r or "Close" in str(r.keys())

    def test_yahoo_offline_fallback_on_error(self):
        """yfinance 失敗時使用離線後備"""
        connector = YahooFinanceConnector()

        def _fail():
            raise RuntimeError("network error")

        with patch.object(connector, "_retry", side_effect=RuntimeError("retry exhausted")):
            # 外層 try/except 會捕獲並用離線後備
            result = list(connector.fetch(symbol="AAPL", period="1mo"))
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("close" in r for r in result)

    def test_eia_fetch_returns_list(self):
        """EIA 無 API key 時使用離線後備"""
        connector = EIAConnector()
        result = list(connector.fetch())
        assert isinstance(result, list)
        if result:
            assert "timestamp" in result[0]
            assert "value" in result[0]

    def test_open_meteo_fetch_returns_list(self):
        """Open-Meteo 無網路時有離線後備"""
        connector = OpenMeteoConnector()
        result = list(connector.fetch(latitude=25.0, longitude=121.0))
        assert isinstance(result, list)
        if result:
            r = result[0]
            assert "timestamp" in r
            assert "temperature_2m" in r or "relative_humidity_2m" in r

    def test_owid_fetch_returns_list(self):
        """OWID 回傳 list[dict]（mock 以避免依賴網路）"""
        connector = OWIDConnector()
        mock_data = {"TW": {"data": [{"date": "2023-01-01", "new_cases": 100, "new_deaths": 5}]}}
        with patch.object(connector, "_request_json", return_value=mock_data):
            result = list(connector.fetch(country_code="TW"))
        assert isinstance(result, list)
        assert len(result) == 1
        assert "timestamp" in result[0]
        assert result[0]["timestamp"] == "2023-01-01"

    def test_owid_offline_when_request_fails(self):
        """網路失敗時使用離線合成序列"""
        connector = OWIDConnector(max_retries=0)
        with patch.object(connector, "_request_json", side_effect=RuntimeError("network")):
            result = list(connector.fetch(country_code="TW"))
        assert len(result) == 100
        assert "new_cases" in result[0]
        assert "timestamp" in result[0]

    def test_newsapi_fetch_returns_list(self):
        """NewsAPI 無 API key 時有離線後備"""
        connector = NewsAPIConnector()
        result = list(connector.fetch(q="market"))
        assert isinstance(result, list)
        if result:
            r = result[0]
            assert "timestamp" in r
            assert "sentiment" in r or "title" in r

    def test_rest_connector_with_mock(self):
        """REST 連接器：mock HTTP 回傳，驗證格式"""
        connector = RESTConnector()

        mock_data = {"items": [{"date": "2023-01-01", "value": 100}, {"date": "2023-01-02", "value": 101}]}

        with patch.object(connector, "_request_json", return_value=mock_data):
            result = list(
                connector.fetch(
                    url="https://api.example.com/data",
                    query={},
                    root_path="items",
                    fields={"timestamp": "date", "value": "value"},
                )
            )
        assert len(result) == 2
        for item in result:
            assert "timestamp" in item
            assert "value" in item

    def test_connector_data_format(self):
        """連接器回傳 flat dict，含 timestamp 與數值欄位"""
        connector = EIAConnector()
        result = list(connector.fetch())
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "timestamp" in item
            for k, v in item.items():
                if k == "timestamp":
                    assert isinstance(v, str)
                else:
                    assert isinstance(v, (int, float, str)) or v is None
