from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from src.api.schemas import CustomerInput, PredictionOutput


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_pipeline.joblib"


def load_model(path=MODEL_PATH):
    if not path.is_file():
        raise RuntimeError(
            f"No se encontró el modelo en {path}. Recuperalo con dvc pull."
        )
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo en {path}.") from exc
    if not all(hasattr(pipeline, name) for name in ("predict", "predict_proba", "feature_names_in_")):
        raise RuntimeError("El archivo no contiene un pipeline compatible para inferencia.")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargo el pipeline una vez, antes de recibir pedidos.
    app.state.pipeline = load_model()
    yield


app = FastAPI(title="Predicción de abandono de clientes", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.pipeline is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput, request: Request):
    pipeline = request.app.state.pipeline
    # Respeto los nombres y el orden usados durante el entrenamiento.
    data = pd.DataFrame([customer.model_dump()], columns=pipeline.feature_names_in_)
    # null debe llegar como NaN al imputador numérico.
    data["TotalCharges"] = data["TotalCharges"].astype(float)
    prediction = int(pipeline.predict(data)[0])
    positive_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(data)[0, positive_index])
    return PredictionOutput(
        prediction="Yes" if prediction == 1 else "No",
        probability=probability,
    )
