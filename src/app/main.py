from typing import List
import uvicorn
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends
from ..utils.model_config_fit import model_config
from .schemas import (
    MushroomFeatures, PredictionResponse, ProbabilityResponse,
    BatchPredictionRequest, BatchPredictionResponse, BatchProbabilityResponse,
    StatusResponse, FitResponse
)
from .config import FEATURE_COLUMNS


app = FastAPI(
    title="Mushroom Classification API",
    description="Определяет, ядовит ли гриб, используя обученную ML-модель.",
    version="1.0.0",
)


def mushroom_to_row(m: MushroomFeatures) -> dict:
    """
    Создаёт словарь всех FEATURE_COLUMNS.
    Меняем только cap-color и habitat,
    остальные оставляем None — OneHotEncoder(handle_unknown='ignore') это поддерживает.
    """
    row = {col: None for col in FEATURE_COLUMNS}
    row["id"] = 0
    row["cap-color"] = m.cap_color
    row["habitat"] = m.habitat
    return row


def mushrooms_to_df(mushrooms: List[MushroomFeatures]) -> pd.DataFrame:
    rows = [mushroom_to_row(m) for m in mushrooms]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


@app.on_event("startup")
def startup():
    loaded = model_config.load_model()


@app.get("/")
async def home():
    return {"message": "Mushroom API работает!"}


@app.get("/predict", response_model=PredictionResponse)
def predict(mushroom: MushroomFeatures = Depends()):
    """
    Предсказание ядовитости гриба
    
    Параметры передаются в query string:
    - cap_color: цвет шляпки (пример: n, b, w, y)
    - habitat: среда обитания (пример: u, g, d)
    
    Пример запроса: /predict?cap_color=n&habitat=u
    """
    if not model_config.is_ready():
        raise HTTPException(500, "Модель не загружена")

    try:
        df = mushrooms_to_df([mushroom])
        pred = int(model_config.model.predict(df)[0])
        label = "poisonous" if pred == 1 else "edible"

        return PredictionResponse(prediction=pred, prediction_label=label)

    except Exception as e:
        raise HTTPException(500, f"Ошибка предсказания: {str(e)}")


@app.get("/predict_proba", response_model=ProbabilityResponse)
def predict_proba(mushroom: MushroomFeatures = Depends()):
    """
    Определение вероятности ядовитости гриба
    
    Возвращает вероятность того, что гриб ядовитый (класс 1),
    а также бинарное предсказание на основе порога 0.5.
    
    Параметры передаются в query string:
    - cap_color: цвет шляпки (пример: n, b, w, y)
    - habitat: среда обитания (пример: u, g, d)
    
    Пример запроса: 
    GET /predict_proba?cap_color=n&habitat=u
    
    Возвращает:
    - probability: вероятность ядовитости (от 0.0 до 1.0)
    - prediction: бинарное предсказание (0 - съедобный, 1 - ядовитый)
    - prediction_label: текстовая метка ('edible' или 'poisonous')
    
    Пример ответа:
    {
        "probability": 0.85,
        "prediction": 1,
        "prediction_label": "poisonous"
    }
    """
    if not model_config.is_ready():
        raise HTTPException(500, "Модель не загружена")

    try:
        df = mushrooms_to_df([mushroom])

        if not hasattr(model_config.model, "predict_proba"):
            raise HTTPException(500, "Модель не поддерживает predict_proba")

        proba_p = float(model_config.model.predict_proba(df)[0][1])
        pred = 1 if proba_p >= 0.5 else 0
        label = "poisonous" if pred == 1 else "edible"

        return ProbabilityResponse(
            probability=proba_p,
            prediction=pred,
            prediction_label=label
        )

    except Exception as e:
        raise HTTPException(500, f"Ошибка: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Пакетное предсказание ядовитости для нескольких грибов
    
    Принимает список грибов и возвращает предсказания для каждого.
    
    Тело запроса (JSON):
    - mushrooms: список объектов с признаками грибов
    
    Пример тела запроса:
    {
        "mushrooms": [
            {"cap_color": "n", "habitat": "u"},
            {"cap_color": "y", "habitat": "g"},
            {"cap_color": "w", "habitat": "d"}
        ]
    }
    
    Возвращает:
    - predictions: список бинарных предсказаний (0 - съедобный, 1 - ядовитый)
    - prediction_labels: список текстовых меток ('edible' или 'poisonous')
    
    Пример ответа:
    {
        "predictions": [1, 0, 1],
        "prediction_labels": ["poisonous", "edible", "poisonous"]
    }
    """
    if not model_config.is_ready():
        raise HTTPException(500, "Модель не загружена")

    try:
        df = mushrooms_to_df(request.mushrooms)
        preds = [int(x) for x in model_config.model.predict(df)]
        labels = ["poisonous" if p == 1 else "edible" for p in preds]

        return BatchPredictionResponse(
            predictions=preds,
            prediction_labels=labels
        )

    except Exception as e:
        raise HTTPException(500, f"Ошибка batch-предсказания: {str(e)}")


@app.post("/predict_proba_batch", response_model=BatchProbabilityResponse)
def predict_proba_batch(request: BatchPredictionRequest):
    """
    Пакетное определение вероятностей ядовитости для нескольких грибов
    
    Принимает список грибов и возвращает вероятности ядовитости для каждого,
    а также бинарные предсказания на основе порога 0.5.
    
    Тело запроса (JSON):
    - mushrooms: список объектов с признаками грибов
    
    Пример тела запроса:
    {
        "mushrooms": [
            {"cap_color": "n", "habitat": "u"},
            {"cap_color": "y", "habitat": "g"},
            {"cap_color": "w", "habitat": "d"}
        ]
    }
    
    Возвращает:
    - probabilities: список вероятностей ядовитости (от 0.0 до 1.0)
    - predictions: список бинарных предсказаний (0 - съедобный, 1 - ядовитый)
    - prediction_labels: список текстовых меток ('edible' или 'poisonous')
    
    Пример ответа:
    {
        "probabilities": [0.85, 0.23, 0.91],
        "predictions": [1, 0, 1],
        "prediction_labels": ["poisonous", "edible", "poisonous"]
    }
    """
    if not model_config.is_ready():
        raise HTTPException(500, "Модель не загружена")

    try:
        df = mushrooms_to_df(request.mushrooms)

        if not hasattr(model_config.model, "predict_proba"):
            raise HTTPException(500, "Модель не поддерживает predict_proba")

        raw_probs = model_config.model.predict_proba(df)[:, 1]

        probabilities = [float(p) for p in raw_probs]
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        labels = ["poisonous" if p == 1 else "edible" for p in predictions]

        return BatchProbabilityResponse(
            probabilities=probabilities,
            predictions=predictions,
            prediction_labels=labels
        )

    except Exception as e:
        raise HTTPException(500, f"Ошибка batch-proba: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def status():
    info = model_config.get_model_info()
    return StatusResponse(
        **info,
        status="готова" if info["is_loaded"] else "не загружена"
    )


@app.post("/fit", response_model=FitResponse)
async def fit():
    """
    Переобучение модели на train_sample.csv
    """
    try:
        df = pd.read_csv("data/train_sample.csv")
        accuracy, trained_at = model_config.fit_from_dataframe(df)

        return FitResponse(
            message=f"Модель успешно обучена ({trained_at})",
            accuracy=accuracy
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8002, reload=True)
