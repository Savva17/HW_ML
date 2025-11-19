from pydantic import BaseModel, Field
from typing import List, Optional


class MushroomFeatures(BaseModel):
    cap_color: str = Field(..., description="Цвет шляпки")
    habitat: str = Field(..., description="Среда обитания")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cap_color": "n",
                "habitat": "u"
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = съедобный, 1 = ядовитый")
    prediction_label: str = Field(..., description="Текстовая метка: 'edible' или 'poisonous'")


class ProbabilityResponse(BaseModel):
    probability: float = Field(..., description="Вероятность, что гриб ядовит")
    prediction: int = Field(..., description="0 = съедобный, 1 = ядовитый")
    prediction_label: str = Field(..., description="Текстовая метка предсказания")


class BatchPredictionRequest(BaseModel):
    mushrooms: List[MushroomFeatures]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mushrooms": [
                        {"cap_color": "n", "habitat": "u"},
                        {"cap_color": "y", "habitat": "g"}
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Предсказания для каждой записи")
    prediction_labels: List[str] = Field(..., description="Текстовые метки для каждого предсказания")


class BatchProbabilityResponse(BaseModel):
    probabilities: List[float] = Field(..., description="Вероятности ядовитости")
    predictions: List[int] = Field(..., description="0/1 для каждого объекта")
    prediction_labels: List[str] = Field(..., description="'edible' или 'poisonous'")


class StatusResponse(BaseModel):
    is_loaded: bool
    model_loaded_date: Optional[str] = None
    model_trained_date: Optional[str] = None
    features: Optional[List[str]] = None
    model_path: Optional[str] = None
    status: Optional[str] = None


class FitResponse(BaseModel):
    message: str
    accuracy: Optional[float]
