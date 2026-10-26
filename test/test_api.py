from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    """Verify that the health check endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_valid_payload():
    """Verify that valid flower features return a successful prediction."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "class_name" in data
    assert data["class_name"] in ["setosa", "versicolor", "virginica"]

def test_predict_out_of_bounds_payload():
    """Verify that out-of-bounds feature values are rejected with a 400 Bad Request."""
    # sepal_length 10.0 exceeds Marc's upper limit of 9.48
    payload = {
        "sepal_length": 10.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
