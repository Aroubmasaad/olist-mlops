
from src.validation import validate_input
from src.config import load_config
from src.features import create_features
from src.preprocessing import load_preprocessor, transform_features
from src.model import load_model, predict
def run_inference(df):
    config = load_config()
    df = validate_input(df)

    features = create_features(df)

    preprocessor = load_preprocessor()
    X = transform_features(features, preprocessor)

    model = load_model()
    predictions, probabilities = predict(model, X)

    return {
        "prediction": predictions.tolist(),
        "probability": probabilities.tolist(),
        "model_version": config["project"]["model_version"],
    }
