import joblib

from src.config import PROJECT_ROOT, load_config
def load_model():
    config = load_config()
    model_path = PROJECT_ROOT / config["paths"]["model"]
    return joblib.load(model_path)
def predict(model, X):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    return prediction, probability
