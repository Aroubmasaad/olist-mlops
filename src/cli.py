import argparse
import time

from src.data import read_input
from src.inference import run_inference
from src.logger import logger


def main(input_path=None):
    if input_path is None:
        parser = argparse.ArgumentParser(
            description="Run late-delivery inference."
        )
        parser.add_argument(
            "--input",
            required=True,
            help="Path to JSON or CSV input file.",
        )
        args = parser.parse_args()
        input_path = args.input

    start_time = time.perf_counter()

    try:
        df = read_input(input_path)

        logger.info(
            "CLI prediction request received | input=%s",
            df.to_dict(orient="records"),
        )

        result = run_inference(df)

    except Exception:
        logger.exception("CLI prediction failed")
        raise

    latency = time.perf_counter() - start_time

    logger.info(
        "CLI prediction completed | prediction=%s | probability=%s | "
        "model_version=%s | latency=%.4fs",
        result["prediction"],
        result["probability"],
        result["model_version"],
        latency,
    )


if __name__ == "__main__":
    main()
