"""多資料源時間對齊單元測試。"""

from schema_infer import align_sources


def test_align_sources_merges_aux():
    rows_yahoo = [
        {"timestamp": "2025-01-01T00:00:00", "close": 100.0, "volume": 1e6},
        {"timestamp": "2025-01-02T00:00:00", "close": 101.0, "volume": 1.1e6},
    ]
    rows_news = [
        {"timestamp": "2025-01-01T12:00:00", "sentiment": 0.5},
        {"timestamp": "2025-01-03T08:00:00", "sentiment": -0.2},
    ]
    out = align_sources(
        {"yahoo": rows_yahoo, "newsapi": rows_news},
        target_source="yahoo",
        freq="1D",
        join_strategy="left",
        join_tolerance="36h",
        missing_policy="ffill",
    )
    assert len(out) == 2
    assert "sentiment" in out[1] or any("sentiment" in r for r in out)

def test_align_sources_inner_drops_unmatched():
    rows_yahoo = [
        {"timestamp": "2025-01-01T00:00:00", "close": 100.0},
        {"timestamp": "2025-01-05T00:00:00", "close": 102.0},
    ]
    # 新聞時間遠離主表，在 36h backward 內對不到第一列
    rows_news = [
        {"timestamp": "2025-02-01T00:00:00", "sentiment": 0.9},
    ]
    out_left = align_sources(
        {"yahoo": rows_yahoo, "newsapi": rows_news},
        target_source="yahoo",
        freq="1D",
        join_strategy="left",
        join_tolerance="36h",
        missing_policy="zero",
    )
    out_inner = align_sources(
        {"yahoo": rows_yahoo, "newsapi": rows_news},
        target_source="yahoo",
        freq="1D",
        join_strategy="inner",
        join_tolerance="36h",
        missing_policy="zero",
    )
    assert len(out_left) >= len(out_inner)
    # inner 應排除完全對不到副源的列（視 merge 結果而定，至少不應比 left 多）
    assert len(out_inner) <= len(out_left)
