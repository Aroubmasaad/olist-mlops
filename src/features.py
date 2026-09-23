import pandas as pd
RAW_FEATURES = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "total_items",
    "total_price",
    "total_freight",
    "total_payment_value",
    "number_of_payments",
    "max_installments",
    "seller_count",
    "distance_km",
    "customer_state",
    "seller_state",
]
NUMERIC_FEATURES = [
    "total_items",
    "total_price",
    "total_freight",
    "total_payment_value",
    "number_of_payments",
    "max_installments",
    "seller_count",
    "distance_km",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "estimated_delivery_days",
    "same_state",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
def create_features(df):
    df = df.copy()
        # Konvertera datum som behövs för feature engineering
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"]
    )
        # Tidsfeatures från ordertillfället
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
        # Planerad leveranstid, känd vid ordertillfället
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
        # Geografisk feature
    df["same_state"] = (
        df["customer_state"] == df["seller_state"]
    ).astype(int)

    return df
