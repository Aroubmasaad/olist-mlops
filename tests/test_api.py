from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200


def test_predict_endpoint():
    order = {
        "order_purchase_timestamp": "2018-01-15T10:30:00",
        "order_estimated_delivery_date": "2018-01-25T10:30:00",
        "total_items": 1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "total_payment_value": 120.0,
        "number_of_payments": 1,
        "max_installments": 1,
        "seller_count": 1,
        "distance_km": 100.0,
        "customer_state": "SP",
        "seller_state": "SP",
    }

    response = client.post("/predict", json=order)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result
    assert "model_version" in result


def test_predict_rejects_invalid_input():
    invalid_order = {
        "order_purchase_timestamp": "2018-01-15T10:30:00",
        "order_estimated_delivery_date": "2018-01-25T10:30:00",
        "total_items": 1,
        "total_price": "wrong_value",
        "total_freight": 20.0,
        "total_payment_value": 120.0,
        "number_of_payments": 1,
        "max_installments": 1,
        "seller_count": 1,
        "distance_km": 100.0,
        "customer_state": "SP",
        "seller_state": "SP",
    }

    response = client.post("/predict", json=invalid_order)

    assert response.status_code == 422


def test_batch_predict_endpoint():
    order = {
        "order_purchase_timestamp": "2018-01-15T10:30:00",
        "order_estimated_delivery_date": "2018-01-25T10:30:00",
        "total_items": 1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "total_payment_value": 120.0,
        "number_of_payments": 1,
        "max_installments": 1,
        "seller_count": 1,
        "distance_km": 100.0,
        "customer_state": "SP",
        "seller_state": "SP",
    }

    response = client.post(
        "/predict/batch",
        json={"orders": [order, order]},
    )

    assert response.status_code == 200

    result = response.json()

    assert "predictions" in result
    assert len(result["predictions"]) == 2
