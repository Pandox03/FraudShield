"""API FraudShield : python -m uvicorn fraudshield.api.main:app --port 8001"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from fraudshield.api.schemas import PredictionOut, Transaction
from fraudshield.config import INFO_PATH, MODEL_PATH
from fraudshield.predict import Predictor

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["predictor"] = Predictor(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="FraudShield",
    version="1.1.0",
    description=(
        "API d'aide à la détection de fraude bancaire. "
        "Les alertes sont indicatives et doivent être validées par un analyste."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/v1/info")
def info_modele():
    if not INFO_PATH.exists():
        raise HTTPException(status_code=404, detail="model_info.json introuvable")
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@app.post("/v1/predict", response_model=PredictionOut)
def predict(transaction: Transaction):
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible. Lancez : python -m fraudshield.train",
        )
    return PredictionOut(**predictor.predict(transaction.model_dump()))


@app.post("/v1/predict/batch")
def predict_batch(transactions: list[Transaction]):
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible.")
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions par appel.")
    return [predictor.predict(t.model_dump()) for t in transactions]
