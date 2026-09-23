import pandas as pd

from src.inference import run_inference


def test_inference_returns_expected_output():
    df = pd.read_csv("artifacts/test.csv").head(1)

    result = run_inference(df)

    assert result["prediction"] == [0]
    assert len(result["probability"]) == 1
    assert result["model_version"] == "1.0.0"