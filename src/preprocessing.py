import joblib
import json
import pandas as pd
from src.features import FEATURE_COLUMNS
from src.config import PROJECT_ROOT, load_config


def load_preprocessor():
    config = load_config()
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
    return joblib.load(preprocessor_path)


def transform_features(df, preprocessor):
    config = load_config()
    feature_list_path = PROJECT_ROOT / config["paths"]["feature_list"]

    with open(feature_list_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    transformed = preprocessor.transform(df[FEATURE_COLUMNS])

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=df.index,
    )
