import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.api.main import MODEL_PATH, load_model
from src.api.schemas import CustomerInput, PredictionOutput
from src.training.train import build_preprocessor


def test_data_schema(customer):
    expected = list(joblib.load(MODEL_PATH).feature_names_in_)
    assert list(CustomerInput.model_fields) == expected
    assert len(expected) == 19
    assert set(CustomerInput.model_json_schema()["required"]) == set(expected)
    assert CustomerInput(**customer).model_dump() == customer
    customer["TotalCharges"] = None
    assert CustomerInput(**customer).TotalCharges is None


@pytest.mark.parametrize("field,value", [
    ("SeniorCitizen", 2), ("SeniorCitizen", True), ("tenure", -1),
    ("tenure", 1.5), ("MonthlyCharges", -1), ("TotalCharges", -1),
    ("MonthlyCharges", float("inf")), ("TotalCharges", float("nan")),
    ("gender", "Unknown"), ("Contract", "Unknown"),
    ("InternetService", "Unknown"), ("PaymentMethod", "Unknown"),
    ("customerID", "123"), ("Churn", "Yes"),
])
def test_schema_rejects_invalid_values(customer, field, value):
    customer[field] = value
    with pytest.raises(ValidationError):
        CustomerInput(**customer)


def test_preprocessor_fit_transform(customer):
    second = dict(customer, TotalCharges=np.nan, tenure=0, Contract="One year")
    data = pd.DataFrame([customer, second])
    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = data.select_dtypes(exclude="number").columns.tolist()
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(data)
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()
    unseen = pd.DataFrame([dict(customer, Contract="Two year", TotalCharges=np.nan)])
    result = preprocessor.transform(unseen)
    assert result.shape == (1, transformed.shape[1])
    assert np.isfinite(result).all()


def test_model_loading():
    pipeline = load_model()
    assert callable(pipeline.predict)
    assert callable(pipeline.predict_proba)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_missing_model(tmp_path):
    with pytest.raises(RuntimeError, match="No se encontró el modelo"):
        load_model(tmp_path / "missing.joblib")


def test_corrupt_model(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"archivo invalido")
    with pytest.raises(RuntimeError, match="No se pudo cargar el modelo"):
        load_model(path)


def test_predict_contract(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "probability"}
    assert result["prediction"] in ("Yes", "No")
    assert 0 <= result["probability"] <= 1
    PredictionOutput(**result)


@pytest.mark.parametrize("change", ["missing", "invalid", "extra"])
def test_invalid_input(client, customer, change):
    if change == "missing":
        del customer["tenure"]
    elif change == "invalid":
        customer["SeniorCitizen"] = 2
    else:
        customer["customerID"] = "123"
    assert client.post("/predict", json=customer).status_code == 422


@pytest.mark.parametrize("total_charges", [600.0, None])
def test_end_to_end(client, customer, total_charges):
    customer["TotalCharges"] = total_charges
    pipeline = joblib.load(MODEL_PATH)
    data = pd.DataFrame([customer], columns=pipeline.feature_names_in_)
    data["TotalCharges"] = data["TotalCharges"].astype(float)
    expected_prediction = "Yes" if pipeline.predict(data)[0] == 1 else "No"
    positive_index = list(pipeline.classes_).index(1)
    expected_probability = pipeline.predict_proba(data)[0, positive_index]
    # El pedido no depende del orden de las claves del JSON.
    response = client.post("/predict", json=dict(reversed(list(customer.items()))))
    assert response.status_code == 200
    assert response.json()["prediction"] == expected_prediction
    assert response.json()["probability"] == pytest.approx(expected_probability)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_docs(client):
    assert client.get("/docs").status_code == 200
