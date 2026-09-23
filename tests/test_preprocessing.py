import numpy as np
import pandas as pd

from src.features import create_features
from src.preprocessing import load_preprocessor, transform_features


def test_preprocessing_matches_notebook_output():
    raw = pd.read_csv("artifacts/test.csv").head(1)

    X_new = transform_features(
        create_features(raw),
        load_preprocessor(),
    )

    X_saved = (
        pd.read_csv("artifacts/model_test.csv")
        .head(1)
        .drop(columns=["late"])
    )

    assert X_new.shape == X_saved.shape
    assert np.allclose(
        X_new.to_numpy(),
        X_saved.to_numpy(),
    )
    