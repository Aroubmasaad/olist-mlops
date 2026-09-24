import pandas as pd

from src.features import RAW_FEATURES, create_features


def test_create_features():
    # Skapar ett litet exempel med en order för att testa feature engineering.
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2018-01-15 10:30:00"],
            "order_estimated_delivery_date": ["2018-01-20 10:30:00"],
            "customer_state": ["SP"],
            "seller_state": ["SP"],
        }
    )

    # Skapar nya features på samma sätt som i inference-pipelinen.
    result = create_features(df)

    # Kontrollerar att datum och geografisk information
    # har omvandlats till rätt features.
    assert result["purchase_month"].iloc[0] == 1
    assert result["purchase_weekday"].iloc[0] == 0
    assert result["purchase_hour"].iloc[0] == 10
    assert result["estimated_delivery_days"].iloc[0] == 5
    assert result["same_state"].iloc[0] == 1


def test_inference_features_do_not_include_leakage():
    # Features som inte får användas vid prediction.
    # "late" är target som modellen ska förutsäga.
    # Leveransdatumen innehåller information som blir känd senare.
    forbidden_features = {
        "late",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
    }

    # Testet misslyckas om någon förbjuden feature finns i RAW_FEATURES.
    assert forbidden_features.isdisjoint(RAW_FEATURES)
