from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SAMPLE = {
    "product_category": "Electronics",
    "product_price_inr": 22055.97,
    "discount_percent": 21.6,
    "expected_delivery_days": 9,
    "actual_delivery_days": 12,
    "customer_rating_preorder": 4.4,
    "previous_returns": 0,
    "previous_orders": 33,
    "payment_method": "Card",
    "damaged_package_flag": "Yes",
    "package_size": "Large",
    "promotion_active": "Yes",
}


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "loaded"}


def test_predict_endpoint() -> None:
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Returned", "Not Returned"}
    assert 0.0 <= body["return_probability"] <= 1.0
    assert body["model"] == "selected-by-held-out-f1"


def test_predict_rejects_invalid_values() -> None:
    response = client.post(
        "/predict", json={**SAMPLE, "discount_percent": 125}
    )
    assert response.status_code == 422


def test_predict_rejects_inconsistent_return_history() -> None:
    response = client.post(
        "/predict", json={**SAMPLE, "previous_returns": 5, "previous_orders": 2}
    )
    assert response.status_code == 422
