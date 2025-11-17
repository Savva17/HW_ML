from pydantic import BaseModel, Field
from typing import List, Optional


class MushroomFeatures(BaseModel):
    """
    Признаки одного гриба из train.csv (кроме целевого столбца 'class').
    Большинство полей необязательные, но 'cap_color' и 'habitat' обязательные.
    """
    cap_diameter: Optional[float] = Field(
        None, description="Диаметр шляпки гриба в сантиметрах"
    )
    cap_shape: Optional[str] = Field(
        None, description="Форма шляпки (например: x – выпуклая, f – плоская)"
    )
    cap_surface: Optional[str] = Field(
        None, description="Тип поверхности шляпки (например: s – гладкая, y – чешуйчатая)"
    )

    cap_color: str = Field(
        ..., description="Цвет шляпки (обязательное поле)"
    )

    does_bruise_or_bleed: Optional[str] = Field(
        None, description="Меняет ли гриб цвет при повреждении или выделяет жидкость"
    )
    gill_attachment: Optional[str] = Field(
        None, description="Способ крепления пластинок"
    )
    gill_spacing: Optional[str] = Field(
        None, description="Частота расположения пластинок"
    )
    gill_color: Optional[str] = Field(
        None, description="Цвет пластинок гриба"
    )

    stem_height: Optional[float] = Field(
        None, description="Высота ножки в сантиметрах"
    )
    stem_width: Optional[float] = Field(
        None, description="Толщина ножки в сантиметрах"
    )
    stem_root: Optional[str] = Field(
        None, description="Тип основания ножки"
    )
    stem_surface: Optional[str] = Field(
        None, description="Поверхность ножки"
    )
    stem_color: Optional[str] = Field(
        None, description="Цвет ножки гриба"
    )

    veil_type: Optional[str] = Field(
        None, description="Тип вуали (частичной или полной)"
    )
    veil_color: Optional[str] = Field(
        None, description="Цвет вуали"
    )
    has_ring: Optional[str] = Field(
        None, description="Есть ли кольцо на ножке (y/n)"
    )
    ring_type: Optional[str] = Field(
        None, description="Тип кольца"
    )
    spore_print_color: Optional[str] = Field(
        None, description="Цвет отпечатка спор"
    )

    habitat: str = Field(
        ..., description="Среда обитания гриба (обязательное поле)"
    )
    season: Optional[str] = Field(
        None, description="Сезон, в котором был найден гриб"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = съедобный, 1 = ядовитый")
    prediction_label: str = Field(..., description="Текстовая метка: 'edible' или 'poisonous'")


class ProbabilityResponse(BaseModel):
    probability: float = Field(..., description="Вероятность, что гриб ядовит")
    prediction: int = Field(..., description="0 = съедобный, 1 = ядовитый")
    prediction_label: str = Field(..., description="Текстовая метка предсказания")


class BatchPredictionRequest(BaseModel):
    mushrooms: List[MushroomFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Предсказания для каждой записи")
    prediction_labels: List[str] = Field(..., description="Текстовые метки для каждого предсказания")


class BatchProbabilityResponse(BaseModel):
    probabilities: List[float] = Field(..., description="Вероятности ядовитости")
    predictions: List[int] = Field(..., description="0/1 для каждого объекта")
    prediction_labels: List[str] = Field(..., description="'edible' или 'poisonous'")


class StatusResponse(BaseModel):
    model_is_loaded: bool
    model_trained_at: Optional[str]
    model_loaded_at: Optional[str]


class FitResponse(BaseModel):
    message: str
    accuracy: Optional[float]
