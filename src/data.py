from pathlib import Path

import pandas as pd


def read_input(path):
    input_path = Path(path)

    if input_path.suffix.lower() == ".json":
        return pd.read_json(input_path)

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(
        "Unsupported file type. Use .json or .csv"
    )