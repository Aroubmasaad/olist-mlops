import pandas as pd
import great_expectations as gx

from src.features import RAW_FEATURES


# Lista över tillåtna delstatskoder som förekommer i datasetet.
# Dessa används för att kontrollera customer_state och seller_state
# innan datan skickas vidare till modellen.
ALLOWED_STATES = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]


def validate_input(df):
    """
    Kontrollerar att alla obligatoriska inputkolumner finns.

    RAW_FEATURES innehåller de kolumner som krävs för att skapa
    de features som modellen behöver. Om en obligatorisk kolumn
    saknas stoppas processen direkt med ett tydligt felmeddelande.
    """

    # Hitta obligatoriska kolumner som saknas i inkommande data.
    missing_columns = [column for column in RAW_FEATURES if column not in df.columns]

    # Om minst en obligatorisk kolumn saknas ska datan inte
    # skickas vidare till feature engineering eller modellen.
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Om alla obligatoriska kolumner finns returneras datan.
    return df


def validate_data_quality(df):
    """
    Validerar kvaliteten på inkommande data innan preprocessing
    och prediktion.

    Valideringen kontrollerar:
    1. Saknade värden i obligatoriska fält.
    2. Datatyp för numeriska värden.
    3. Rimliga numeriska intervall med Great Expectations.
    4. Tillåtna delstatskoder.
    5. Giltiga datum och logisk ordning mellan datumen.

    Felaktig data stoppas innan den når ML-modellen.
    """

    # ---------------------------------------------------------
    # 1. KONTROLL AV SAKNADE VÄRDEN
    # ---------------------------------------------------------

    # Dessa fält måste alltid ha ett värde eftersom de behövs
    # för feature engineering och preprocessing.
    required_non_null_columns = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "customer_state",
        "seller_state",
    ]

    # Kontrollera missing ratio för varje obligatoriskt fält.
    # För dessa kolumner är den tillåtna missing-ration 0 %.
    for column in required_non_null_columns:
        missing_ratio = df[column].isna().mean()

        if missing_ratio > 0:
            raise ValueError(
                f"Data validation failed: {column} has a missing ratio "
                f"of {missing_ratio:.2%}. Allowed missing ratio is 0%."
            )

    # Numeriska saknade värden stoppas inte här.
    # Den sparade preprocessorn från träningen innehåller en imputer
    # som hanterar numeriska saknade värden på samma sätt som i Task 2.

    # ---------------------------------------------------------
    # 2. KONTROLL AV NUMERISKA DATATYPER
    # ---------------------------------------------------------

    # Dessa kolumner ska innehålla numeriska värden.
    numeric_columns = [
        "total_items",
        "total_price",
        "total_freight",
        "total_payment_value",
        "number_of_payments",
        "max_installments",
        "seller_count",
        "distance_km",
    ]

    for column in numeric_columns:
        # Saknade numeriska värden tas bort endast under denna kontroll,
        # eftersom de senare kan hanteras av den sparade imputern.
        non_null_values = df[column].dropna()

        # Försök tolka alla befintliga värden som numeriska.
        # errors="coerce" gör ogiltiga värden, t.ex. "hello", till NaN.
        numeric_values = pd.to_numeric(
            non_null_values,
            errors="coerce",
        )

        # Om ett befintligt värde inte kunde tolkas som ett tal
        # stoppas datan innan den når modellen.
        if not numeric_values.notna().all():
            raise ValueError(
                f"Data validation failed: {column} must contain numeric values."
            )

    # ---------------------------------------------------------
    # 3. SKAPA EN GREAT EXPECTATIONS-BATCH
    # ---------------------------------------------------------

    # Ett tillfälligt Great Expectations-context skapas i minnet.
    # "ephemeral" betyder att vi inte behöver skapa en permanent
    # Great Expectations-konfiguration på disk för denna validering.
    context = gx.get_context(mode="ephemeral")

    # Skapa en pandas-datakälla för inkommande DataFrame.
    data_source = context.data_sources.add_pandas(name="inference_source")

    # Registrera inkommande orderdata som en data asset.
    data_asset = data_source.add_dataframe_asset(name="inference_orders")

    # Definiera att hela inkommande DataFrame ska valideras.
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "inference_batch"
    )

    # Skapa den batch som Great Expectations ska kontrollera.
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # ---------------------------------------------------------
    # 4. KONTROLL AV NUMERISKA INTERVALL
    # ---------------------------------------------------------

    # En order måste innehålla minst en produkt.
    total_items_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_items",
        min_value=1,
    )

    if not batch.validate(total_items_expectation).success:
        raise ValueError("Data validation failed: total_items must be at least 1.")

    # Orderpriset får inte vara negativt.
    price_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_price",
        min_value=0,
    )

    if not batch.validate(price_expectation).success:
        raise ValueError("Data validation failed: total_price cannot be negative.")

    # Fraktkostnaden får inte vara negativ.
    freight_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_freight",
        min_value=0,
    )

    if not batch.validate(freight_expectation).success:
        raise ValueError("Data validation failed: total_freight cannot be negative.")

    # Det totala betalningsvärdet får inte vara negativt.
    payment_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_payment_value",
        min_value=0,
    )

    if not batch.validate(payment_expectation).success:
        raise ValueError(
            "Data validation failed: total_payment_value cannot be negative."
        )

    # Antalet betalningar måste vara minst 1.
    payments_count_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="number_of_payments",
        min_value=1,
    )

    if not batch.validate(payments_count_expectation).success:
        raise ValueError(
            "Data validation failed: number_of_payments must be at least 1."
        )

    # Maximalt antal avbetalningar måste vara minst 1.
    installments_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="max_installments",
        min_value=1,
    )

    if not batch.validate(installments_expectation).success:
        raise ValueError("Data validation failed: max_installments must be at least 1.")

    # En order måste ha minst en säljare.
    seller_count_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="seller_count",
        min_value=1,
    )

    if not batch.validate(seller_count_expectation).success:
        raise ValueError("Data validation failed: seller_count must be at least 1.")

    # Avståndet mellan kund och säljare får inte vara negativt.
    distance_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="distance_km",
        min_value=0,
    )

    if not batch.validate(distance_expectation).success:
        raise ValueError("Data validation failed: distance_km cannot be negative.")

    # ---------------------------------------------------------
    # 5. KONTROLL AV TILLÅTNA DELSTATSKODER
    # ---------------------------------------------------------

    # Kontrollera att kundens delstatskod finns i ALLOWED_STATES.
    customer_state_expectation = gx.expectations.ExpectColumnValuesToBeInSet(
        column="customer_state",
        value_set=ALLOWED_STATES,
    )

    if not batch.validate(customer_state_expectation).success:
        raise ValueError(
            "Data validation failed: customer_state is not an allowed state code."
        )

    # Kontrollera även säljarens delstatskod.
    seller_state_expectation = gx.expectations.ExpectColumnValuesToBeInSet(
        column="seller_state",
        value_set=ALLOWED_STATES,
    )

    if not batch.validate(seller_state_expectation).success:
        raise ValueError(
            "Data validation failed: seller_state is not an allowed state code."
        )

    # ---------------------------------------------------------
    # 6. KONTROLL AV DATUM
    # ---------------------------------------------------------

    try:
        # Konvertera köpdatumet till datetime.
        # errors="raise" betyder att ogiltiga datum ska ge ett fel.
        purchase_date = pd.to_datetime(
            df["order_purchase_timestamp"],
            errors="raise",
        )

        # Konvertera beräknat leveransdatum på samma sätt.
        estimated_date = pd.to_datetime(
            df["order_estimated_delivery_date"],
            errors="raise",
        )

    except (ValueError, TypeError):
        # Om ett datum inte kan tolkas stoppas datan.
        raise ValueError(
            "Data validation failed: date columns must contain valid dates."
        )

    # Det beräknade leveransdatumet måste ligga efter köpdatumet.
    # Ett leveransdatum före eller samma dag/tid som köpdatumet
    # betraktas här som ogiltig input.
    if (estimated_date <= purchase_date).any():
        raise ValueError(
            "Data validation failed: order_estimated_delivery_date "
            "must be after order_purchase_timestamp."
        )

    # Om alla kontroller lyckas är datan godkänd
    # och kan skickas vidare till feature engineering,
    # preprocessing och slutligen ML-modellen.
    return df


# Jag validerar inkommande data innan den skickas till ML-modellen.
# Först kontrollerar jag att alla obligatoriska kolumner finns.
# Sedan kontrollerar jag saknade värden, numeriska datatyper, rimliga intervall, tillåtna kategorier och datum.
# Jag använder Great Expectations för datakvalitetsreglerna. Syftet är att felaktig data ska stoppas innan feature engineering, preprocessing och prediktion.
# Numeriska saknade värden kan hanteras av den redan tränade och sparade imputern, så jag tränar inte om preprocessorn under inference.
