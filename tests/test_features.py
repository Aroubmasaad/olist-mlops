import pandas as pd

from src.features import create_features


def test_create_features():
    df = pd.DataFrame({
        "order_purchase_timestamp": ["2018-01-15 10:30:00"],
        "order_estimated_delivery_date": ["2018-01-20 10:30:00"],
        "customer_state": ["SP"],
        "seller_state": ["SP"],
    })

    result = create_features(df)

    assert result["purchase_month"].iloc[0] == 1
    assert result["purchase_weekday"].iloc[0] == 0
    assert result["purchase_hour"].iloc[0] == 10
    assert result["estimated_delivery_days"].iloc[0] == 5
    assert result["same_state"].iloc[0] == 1
    