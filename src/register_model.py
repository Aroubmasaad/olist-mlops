import json
from pathlib import Path

import joblib
import mlflow


# ---------------------------------------------------------
# Sökvägar
# ---------------------------------------------------------

# Hitta projektets rotmapp automatiskt.
# __file__ är denna fil: src/register_model.py
# parents[1] går upp till projektmappen: olist_MLops
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Sökväg till den redan tränade modellen från Task 2.
MODEL_PATH = PROJECT_ROOT / "artifacts" / "late_delivery_model.joblib"

# Sökväg till resultaten som skapades i Notebook 6.
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "model_results.json"


# ---------------------------------------------------------
# Läs modellresultat från Task 2
# ---------------------------------------------------------

# Läs parametrar och metrics från JSON-filen.
# Vi använder de sparade resultaten i stället för att skriva
# värdena manuellt i denna fil.
with open(RESULTS_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)


# ---------------------------------------------------------
# Ladda den redan tränade modellen
# ---------------------------------------------------------

# Modellen tränades och sparades i Task 2.
# Här laddar vi endast modellen.
# Vi använder INTE model.fit(), eftersom ingen ny träning
# ska göras i Task 3.
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Skapa/välj MLflow Experiment
# ---------------------------------------------------------

# Ett Experiment samlar relaterade MLflow Runs.
# Detta experiment används för projektets late-delivery-modell.
mlflow.set_experiment("olist-late-delivery")


# ---------------------------------------------------------
# Starta ett MLflow Run
# ---------------------------------------------------------

# Ett Run innehåller information om en specifik modell:
# parameters, metrics och artifacts.
with mlflow.start_run(run_name="selected-logistic-regression"):

    # -----------------------------------------------------
    # Logga modellens parametrar
    # -----------------------------------------------------

    # Parametrarna läses från model_results.json.
    # Exempel:
    # model = LogisticRegression
    # C = 10
    # class_weight = balanced
    mlflow.log_param("model", results["model"])
    mlflow.log_param("C", results["C"])
    mlflow.log_param("class_weight", results["class_weight"])
    mlflow.log_param("primary_metric", results["primary_metric"])

    # -----------------------------------------------------
    # Logga validation metrics
    # -----------------------------------------------------

    # Gå igenom alla metrics för den valda modellen
    # på validation-datasetet och spara dem i MLflow.
    #
    # Exempel:
    # validation_accuracy
    # validation_recall
    # validation_average_precision
    for metric_name, metric_value in results["selected_model_validation"].items():
        mlflow.log_metric(
            f"validation_{metric_name}",
            metric_value,
        )

    # -----------------------------------------------------
    # Logga test metrics
    # -----------------------------------------------------

    # Testresultaten loggas separat så att validation-
    # och testresultat inte blandas ihop.
    #
    # Exempel:
    # test_accuracy
    # test_recall
    # test_average_precision
    for metric_name, metric_value in results["final_test"].items():
        mlflow.log_metric(
            f"test_{metric_name}",
            metric_value,
        )

    # -----------------------------------------------------
    # Logga och registrera modellen
    # -----------------------------------------------------

    # Vi registrerar samma modell som tränades i Task 2.
    #
    # sk_model:
    #   Den redan laddade LogisticRegression-modellen.
    #
    # name:
    #   Namnet på modellen som artifact i detta MLflow Run.
    #
    # registered_model_name:
    #   Namnet som modellen får i MLflow Model Registry.
    #
    # Första registreringen skapar normalt Version 1.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="olist-late-delivery-model",
    )