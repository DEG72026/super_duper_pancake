import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier

import api.app as api_app
from api.app import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def test_model():
    """Give the API a small tree trained on data/iris.csv instead of the bucket model."""
    data = np.loadtxt("data/iris.csv", delimiter=",", skiprows=1)
    api_app.model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(data[:, :4], data[:, 4].astype(int))
    yield
    api_app.model = None


def test_health_check():
    """Verify that the health check endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_payload():
    """Verify that each species is predicted correctly for a typical flower."""
    examples = [
        ((5.1, 3.5, 1.4, 0.2), "setosa"),
        ((6.0, 2.9, 4.5, 1.5), "versicolor"),
        ((6.9, 3.1, 5.4, 2.1), "virginica"),
    ]
    for (sl, sw, pl, pw), name in examples:
        payload = {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["class_name"] == name


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
    assert "detail" in response.json()


def test_without_model_health_is_degraded_and_predict_returns_503():
    """Without a model the API says so instead of guessing."""
    api_app.model = None
    assert client.get("/health").json()["status"] == "degraded"
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict", json=payload).status_code == 503
