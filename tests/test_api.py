import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.utils.model_config_fit import model_config


client = TestClient(app)


def test_home():
    """GET / — проверяем, что API работает."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    """GET /predict — базовое предсказание."""
    model_config.load_model()
    
    if not model_config.is_ready():
        pytest.skip("Модель не загружена, пропускаем тест")
    
    params = {"cap_color": "n", "habitat": "u"}
    response = client.get("/predict", params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "prediction_label" in data
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["edible", "poisonous"]


def test_predict_proba():
    """GET /predict_proba — вероятность ядовитости."""
    params = {"cap_color": "y", "habitat": "g"}
    response = client.get("/predict_proba", params=params)

    assert response.status_code == 200
    data = response.json()

    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert data["prediction"] in [0, 1]


def test_status():
    """GET /status — проверяем статус модели."""
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()

    assert "is_loaded" in data
    assert "model_loaded_date" in data


def test_openapi_available():
    """GET /openapi.json — схема API доступна."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "paths" in data
    assert "/predict" in data["paths"]
    assert "/status" in data["paths"]
