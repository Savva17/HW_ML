import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd

from ..utils.model_config_fit import model_config

from .schemas import (
    MushroomFeatures, PredictionResponse, ProbabilityResponse,
    BatchPredictionRequest, BatchPredictionResponse, BatchProbabilityResponse,
    StatusResponse, FitResponse
)


app = FastAPI(
    title="Mushroom Classification API",
    description="Определяет, ядовит ли гриб, используя обученную ML-модель.",
    version="1.0.0",
)


# ---------------------- STARTUP ----------------------
@app.on_event("startup")
async def startup_event():
    model_config.load_model()


@app.get("/")
async def root():
    return {"message": "Mushroom Classification API работает!"}


# ---------------------- Вспомогательная функция ----------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Кодирует признаки гриба с помощью сохранённых LabelEncoder."""
    encoded = df.copy()

    for col in encoded.columns:
        if col in model_config.label_encoders:
            le = model_config.label_encoders[col]
            encoded[col] = encoded[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    return encoded


# ---------------------- PREDICT ----------------------
@app.get("/predict", response_model=PredictionResponse)
async def predict(features: MushroomFeatures):
    if not model_config.is_ready():
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        df = pd.DataFrame([features.dict()])
        df = encode_features(df)

        pred = model_config.model.predict(df)[0]

        return PredictionResponse(
            prediction=int(pred),
            prediction_label="ядовитый" if pred == 1 else "съедобный"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


# ---------------------- PREDICT PROBA ----------------------
@app.get("/predict_proba", response_model=ProbabilityResponse)
async def predict_proba(features: MushroomFeatures):
    if not model_config.is_ready():
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        df = pd.DataFrame([features.dict()])
        df = encode_features(df)

        proba = model_config.model.predict_proba(df)[0][1]
        pred = 1 if proba > 0.5 else 0

        return ProbabilityResponse(
            probability=float(proba),
            prediction=pred,
            prediction_label="ядовитый" if pred == 1 else "съедобный"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


# ---------------------- PREDICT BATCH ----------------------
@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model_config.is_ready():
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        df = pd.DataFrame([m.dict() for m in request.mushrooms])
        df = encode_features(df)

        preds = model_config.model.predict(df)

        return BatchPredictionResponse(
            predictions=[int(p) for p in preds],
            prediction_labels=["ядовитый" if p == 1 else "съедобный" for p in preds]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка batch-предсказания: {str(e)}")


# ---------------------- PREDICT PROBA BATCH ----------------------
@app.post("/predict_proba_batch", response_model=BatchProbabilityResponse)
async def predict_proba_batch(request: BatchPredictionRequest):
    if not model_config.is_ready():
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        df = pd.DataFrame([m.dict() for m in request.mushrooms])
        df = encode_features(df)

        probs = model_config.model.predict_proba(df)[:, 1]
        preds = [1 if p > 0.5 else 0 for p in probs]

        return BatchProbabilityResponse(
            probabilities=[float(p) for p in probs],
            predictions=preds,
            prediction_labels=["ядовитый" if p == 1 else "съедобный" for p in preds]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка batch-proba: {str(e)}")


# ---------------------- STATUS ----------------------
@app.get("/status", response_model=StatusResponse)
async def status():
    info = model_config.get_model_info()
    return StatusResponse(
        model_loaded_date=info["model_loaded_date"],
        status="готова" if info["is_loaded"] else "не загружена",
    )


# ---------------------- FIT ----------------------
@app.post("/fit", response_model=FitResponse)
async def fit():
    """
    Переобучение модели на train.csv
    """
    try:
        df = pd.read_csv("data/train.csv")
        accuracy, trained_at = model_config.fit_from_dataframe(df)

        return FitResponse(
            message=f"Модель успешно обучена ({trained_at})",
            accuracy=accuracy
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8001, reload=True)
