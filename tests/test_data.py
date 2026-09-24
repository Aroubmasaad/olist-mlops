from src.data import read_input


def test_read_json_input():
    df = read_input("data/sample_order.json")

    assert df.shape == (1, 12)
