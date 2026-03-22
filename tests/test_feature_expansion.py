import numpy as np

from feature_expansion import expand_timeseries_features, merge_expansion_config


def test_expand_adds_columns():
    X = np.arange(20, dtype=float).reshape(10, 2)
    names = ["close", "volume"]
    X2, n2 = expand_timeseries_features(
        X,
        names,
        lag_steps=(1,),
        rolling_windows=(3,),
    )
    assert X2.shape[0] == 10
    assert X2.shape[1] > 2
    assert any("lag1" in x for x in n2)
    assert "close" in n2


def test_merge_expansion_config():
    assert merge_expansion_config(None, False) is None
    assert merge_expansion_config({"enabled": True}, False) is not None
    assert merge_expansion_config({"enabled": False}, True) is not None
    assert merge_expansion_config({"enabled": False}, True)["enabled"]
