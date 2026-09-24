import os
import mlflow
from src.config import load_config


def load_model():
    """
    Laddar den registrerade modellen från MLflow Model Registry.

    Modellen tränas inte här.
    Den tränades redan i Task 2 och registrerades i MLflow.
    """

    # Läs projektets konfiguration från config.yaml.
    config = load_config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Hämta modellnamn och modellversion från MLflow-konfigurationen.
    model_name = config["mlflow"]["model_name"]
    model_version = config["mlflow"]["model_version"]

    # Skapa adressen (URI) till modellen i MLflow Model Registry.
    # Exempel:
    # models:/olist-late-delivery-model/1
    model_uri = f"models:/{model_name}/{model_version}"

    # Ladda den redan registrerade modellen från MLflow.
    # Ingen model.fit() eller ny träning görs här.
    model = mlflow.sklearn.load_model(model_uri)

    return model


def predict(model, X):
    """
    Gör prediction och returnerar klass och sannolikhet.
    """

    # 0 = on time
    # 1 = late
    prediction = model.predict(X)

    # Sannolikheten för klass 1 = late.
    probability = model.predict_proba(X)[:, 1]

    return prediction, probability
