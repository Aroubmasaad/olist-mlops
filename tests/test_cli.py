from src.cli import main


def test_cli_runs():
    main("data/sample_order.json")
