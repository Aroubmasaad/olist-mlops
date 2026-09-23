import pandas as pd
import pytest

from src.validation import validate_input, validate_data_quality


def test_missing_required_column_raises_error():
    df = pd.read_csv("artifacts/test.csv").head(1)
    df = df.drop(columns=["distance_km"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)
def get_valid_input():
    """
    Skapar ett exempel på giltig inputdata.

    Funktionen används som grund i testerna nedan.
    Varje test ändrar sedan en specifik sak i datan för att kontrollera
    att valideringen upptäcker felet.
    """
    return pd.DataFrame({
        "order_purchase_timestamp": ["2018-01-10"],
        "order_estimated_delivery_date": ["2018-01-20"],
        "total_items": [2],
        "total_price": [100],
        "total_freight": [10],
        "total_payment_value": [110],
        "number_of_payments": [2],
        "max_installments": [2],
        "seller_count": [1],
        "distance_km": [100],
        "customer_state": ["SP"],
        "seller_state": ["RJ"],
    })


def test_negative_price_is_rejected():
    """
    Testar att ett negativt orderpris inte accepteras.

    total_price ska vara 0 eller större.
    Här sätter vi medvetet ett felaktigt värde (-10)
    och kontrollerar att valideringen stoppar datan.
    """
    df = get_valid_input()
    df["total_price"] = -10

    with pytest.raises(
        ValueError,
        match="total_price cannot be negative",
    ):
        validate_data_quality(df)


def test_invalid_numeric_type_is_rejected():
    """
    Testar att numeriska kolumner verkligen innehåller numeriska värden.

    Här ersätter vi total_price med texten "hello".
    Eftersom modellen förväntar sig ett numeriskt värde
    ska datavalideringen avvisa denna input.
    """
    df = get_valid_input()
    df["total_price"] = "hello"

    with pytest.raises(
        ValueError,
        match="total_price must contain numeric values",
    ):
        validate_data_quality(df)


def test_invalid_state_is_rejected():
    """
    Testar att customer_state innehåller en tillåten delstatskod.

    "SP" och "RJ" är exempel på giltiga koder i datasetet.
    Här använder vi medvetet "XYZ", som inte finns i listan
    över tillåtna delstatskoder.
    """
    df = get_valid_input()
    df["customer_state"] = "XYZ"

    with pytest.raises(
        ValueError,
        match="customer_state is not an allowed state code",
    ):
        validate_data_quality(df)


def test_missing_required_value_is_rejected():
    """
    Testar att obligatoriska fält inte får sakna värden.

    customer_state behövs för feature engineering och preprocessing.
    Därför ska ett saknat värde (None/NaN) i detta fält avvisas.
    """
    df = get_valid_input()
    df.loc[0, "customer_state"] = None

    with pytest.raises(
        ValueError,
        match="customer_state cannot contain missing values",
    ):
        validate_data_quality(df)


def test_invalid_delivery_date_is_rejected():
    """
    Testar den logiska ordningen mellan köpdatum och beräknat leveransdatum.

    Det beräknade leveransdatumet måste ligga efter köpdatumet.
    Här skapar vi medvetet felaktig data där leveransdatumet
    ligger före köpdatumet.
    """
    df = get_valid_input()
    df["order_purchase_timestamp"] = "2018-01-20"
    df["order_estimated_delivery_date"] = "2018-01-10"

    with pytest.raises(
        ValueError,
        match=(
            "order_estimated_delivery_date must be "
            "after order_purchase_timestamp"
        ),
    ):
        validate_data_quality(df)