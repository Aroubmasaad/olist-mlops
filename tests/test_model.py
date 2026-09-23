import pandas as pd

from src.features import create_features
from src.preprocessing import load_preprocessor, transform_features
from src.model import load_model, predict


def test_model_loads_and_predicts():
    raw = pd.read_csv("artifacts/test.csv").head(1)

    features = create_features(raw)
    preprocessor = load_preprocessor()
    X = transform_features(features, preprocessor)

    model = load_model()
    predictions, probabilities = predict(model, X)

    assert len(predictions) == 1
    assert len(probabilities) == 1
    assert predictions[0] in [0, 1]
    assert 0 <= probabilities[0] <= 1
