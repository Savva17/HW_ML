import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime


class ModelConfig:
    """
    Класс для управления жизненным циклом ML-модели:
    - загрузка модели;
    - обучение модели;
    - хранение информации о модели;
    - предоставление статуса модели.

    Модель сохраняется в формате .pkl и представляет собой sklearn Pipeline
    с препроцессором (OneHotEncoder) и классификатором RandomForestClassifier.
    """
    def __init__(self, model_path: str = "mushroom_model.pkl") -> None:
        self.model_path = model_path
        self.model: Pipeline | None = None
        self.features: list[str] | None = None
        self.model_loaded_date: str | None = None
        self.model_trained_date: str | None = None

    def load_model(self) -> bool:
        """
        Загружает обученную модель из файла .pkl.
        """
        if not os.path.exists(self.model_path):
            print(f"Файл модели не найден: {self.model_path}")
            return False
        try:
            self.model = joblib.load(self.model_path)
            self.model_loaded_date = datetime.now().isoformat()
            ts = os.path.getmtime(self.model_path)
            self.model_trained_date = datetime.fromtimestamp(ts).isoformat()
            print("✅ Модель успешно загружена!")
            return True
        except Exception as e:
            print(f"❌ ВНИМАНИЕ: модель не загружена! Ошибка: {str(e)}")
            return False

    def fit_from_dataframe(self, df: pd.DataFrame, target_column: str = "class") -> tuple[float, str]:
        """
        Обучает модель заново на переданном DataFrame.
        """
        if target_column not in df.columns:
            raise ValueError(f"Нет столбца '{target_column}'")

        X = df.drop(columns=[target_column], errors="ignore")
        y_raw = df[target_column]
        y = (y_raw == "p").astype(int)

        categorical_features = list(X.columns)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        acc = float(model.score(X_test, y_test))
        now_iso = datetime.now().isoformat()

        joblib.dump(model, self.model_path)

        self.model = model
        self.features = categorical_features
        self.model_trained_date = now_iso
        self.model_loaded_date = now_iso

        return acc, now_iso

    def is_ready(self) -> bool:
        """
        Проверяет, загружена ли модель.
        """
        return self.model is not None

    def get_model_info(self) -> dict:
        """
        Возвращает информацию о текущей модели.

        Returns:
            dict: информация о модели, включая:
                - is_loaded: bool — загружена ли модель
                - model_loaded_date: str — дата последней загрузки
                - model_trained_date: str — дата последней тренировки
                - features: list[str] — список признаков
                - model_path: str — путь к файлу модели
        """
        return {
            "is_loaded": self.model is not None,
            "model_loaded_date": self.model_loaded_date,
            "model_trained_date": self.model_trained_date,
            "features": self.features,
            "model_path": self.model_path,
        }


model_config = ModelConfig()
