import pandas as pd
import pytest

from src.validation import validate_input, validate_data_quality


# ---------------------------------------------------------
# 1. TEST AV SAKNAD OBLIGATORISK KOLUMN
# ---------------------------------------------------------

def test_missing_required_column_raises_error():
    """
    Testar att validate_input stoppar data om en obligatorisk
    input-kolumn saknas.

    Här läser vi en order från testdatan och tar medvetet bort
    kolumnen distance_km.

    Eftersom distance_km finns i RAW_FEATURES ska validate_input
    upptäcka att kolumnen saknas och skapa ett ValueError.
    """

    # Läs endast den första raden från testdatan.
    df = pd.read_csv("artifacts/test.csv").head(1)

    # Ta medvetet bort en obligatorisk input-kolumn.
    df = df.drop(columns=["distance_km"])

    # Vi förväntar oss att validate_input stoppar datan.
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_input(df)


# ---------------------------------------------------------
# 2. HJÄLPFUNKTION SOM SKAPAR GILTIG TESTDATA
# ---------------------------------------------------------

def get_valid_input():
    """
    Skapar ett exempel på giltig inputdata.

    Denna funktion används som grund i testerna nedan.

    I varje test börjar vi alltså med korrekt data och ändrar
    sedan medvetet en specifik sak för att kontrollera att
    valideringen upptäcker felet.
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


# ---------------------------------------------------------
# 3. TEST AV NEGATIVT PRIS
# ---------------------------------------------------------

def test_negative_price_is_rejected():
    """
    Testar att ett negativt orderpris inte accepteras.

    total_price ska vara 0 eller större.

    Här sätter vi medvetet total_price till -10.
    Great Expectations-regeln i validate_data_quality
    ska upptäcka detta och stoppa datan.
    """

    # Börja med giltig input.
    df = get_valid_input()

    # Skapa medvetet ett felaktigt negativt pris.
    df["total_price"] = -10

    # Valideringen ska skapa ett ValueError.
    with pytest.raises(
        ValueError,
        match="total_price cannot be negative",
    ):
        validate_data_quality(df)


# ---------------------------------------------------------
# 4. TEST AV FELAKTIG NUMERISK DATATYP
# ---------------------------------------------------------

def test_invalid_numeric_type_is_rejected():
    """
    Testar att numeriska kolumner verkligen innehåller
    numeriska värden.

    total_price ska innehålla ett tal.

    Här använder vi texten "hello" i stället för ett numeriskt
    värde. Valideringen ska därför stoppa datan innan den når
    preprocessing och ML-modellen.
    """

    # Börja med giltig input.
    df = get_valid_input()

    # Lägg medvetet in text i en numerisk kolumn.
    df["total_price"] = "hello"

    # Valideringen ska upptäcka den felaktiga datatypen.
    with pytest.raises(
        ValueError,
        match="total_price must contain numeric values",
    ):
        validate_data_quality(df)


# ---------------------------------------------------------
# 5. TEST AV OGILTIG DELSTATSKOD
# ---------------------------------------------------------

def test_invalid_state_is_rejected():
    """
    Testar att customer_state innehåller en tillåten
    delstatskod.

    Exempelvis SP och RJ är giltiga koder.

    Här använder vi medvetet XYZ, som inte finns i
    ALLOWED_STATES.

    Great Expectations ska därför upptäcka den ogiltiga
    kategorin och stoppa datan.
    """

    # Börja med giltig input.
    df = get_valid_input()

    # Skapa medvetet en ogiltig kategori.
    df["customer_state"] = "XYZ"

    # Valideringen ska stoppa den ogiltiga delstatskoden.
    with pytest.raises(
        ValueError,
        match="customer_state is not an allowed state code",
    ):
        validate_data_quality(df)


# ---------------------------------------------------------
# 6. TEST AV SAKNAT VÄRDE I OBLIGATORISKT FÄLT
# ---------------------------------------------------------

def test_missing_required_value_is_rejected():
    """
    Testar att obligatoriska fält inte får sakna värden.

    customer_state behövs för feature engineering och
    preprocessing.

    För obligatoriska datum- och kategorifält använder
    projektet en tillåten missing ratio på 0 %.

    Eftersom detta exempel endast innehåller en order betyder
    ett saknat customer_state att missing ratio blir 100 %.
    Datan ska därför stoppas.
    """

    # Börja med giltig input.
    df = get_valid_input()

    # Ta medvetet bort customer_state.
    df.loc[0, "customer_state"] = None

    # Valideringen ska upptäcka missing ratio och stoppa datan.
    with pytest.raises(
        ValueError,
        match="customer_state has a missing ratio",
    ):
        validate_data_quality(df)


# ---------------------------------------------------------
# 7. TEST AV FELAKTIG DATUMORDNING
# ---------------------------------------------------------

def test_invalid_delivery_date_is_rejected():
    """
    Testar den logiska ordningen mellan köpdatum och
    beräknat leveransdatum.

    order_estimated_delivery_date måste ligga efter
    order_purchase_timestamp.

    Här skapar vi medvetet felaktig data där det beräknade
    leveransdatumet ligger före köpdatumet.
    """

    # Börja med giltig input.
    df = get_valid_input()

    # Köpdatum sätts efter det beräknade leveransdatumet.
    df["order_purchase_timestamp"] = "2018-01-20"
    df["order_estimated_delivery_date"] = "2018-01-10"

    # Valideringen ska stoppa den logiskt felaktiga datan.
    with pytest.raises(
        ValueError,
        match=(
            "order_estimated_delivery_date must be "
            "after order_purchase_timestamp"
        ),
    ):
        validate_data_quality(df)


# ---------------------------------------------------------
# 8. TEST AV MISSING RATIO I EN BATCH
# ---------------------------------------------------------

def test_missing_ratio_in_required_column_is_rejected():
    """
    Testar projektets missing-ratio-policy med flera orders.

    Vi skapar fyra giltiga orders och tar sedan bort
    customer_state från en av dem.

    1 saknat värde av 4 orders ger:

        1 / 4 = 0.25 = 25 %

    För obligatoriska kategorifält är den tillåtna
    missing-ration 0 %.

    Därför ska en batch med 25 % missing customer_state
    stoppas innan den når ML-modellen.
    """

    # Skapa först en giltig order.
    df = get_valid_input()

    # Kopiera samma order fyra gånger så att vi får en batch
    # med totalt fyra orders.
    df = pd.concat(
        [df] * 4,
        ignore_index=True,
    )

    # Ta bort customer_state från den första ordern.
    # Nu saknas värdet i 1 av 4 orders = 25 % missing.
    df.loc[0, "customer_state"] = None

    # Eftersom tillåten missing ratio är 0 % ska
    # validate_data_quality stoppa denna batch.
    with pytest.raises(
        ValueError,
        match="missing ratio",
    ):
        validate_data_quality(df)