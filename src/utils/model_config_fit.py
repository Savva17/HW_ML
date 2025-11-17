import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ModelConfig:
    """Конфигурация и управление моделью (загрузка, обучение, информация)."""

    def __init__(self, model_path: str = "mushroom_model.pkl") -> None:
        self.model_path: str = model_path

        self.model: Optional[RandomForestClassifier] = None
        self.label_encoders: Optional[Dict[str, LabelEncoder]] = None
        self.features: Optional[List[str]] = None

        self.model_loaded_date: Optional[str] = None
        self.model_trained_date: Optional[str] = None

    def load_model(self) -> bool:
        """Загрузка модели из файла pkl."""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Файл модели не найден: {self.model_path}")
                return False

            model_data = joblib.load(self.model_path)

            self.model = model_data["model"]
            self.label_encoders = model_data["label_encoders"]
            # список признаков сохраняем в pkl при обучении
            self.features = model_data.get("features")

            self.model_loaded_date = datetime.now().isoformat()
            # если при обучении мы сохраняем дату – прочитаем её
            self.model_trained_date = model_data.get("model_trained_date")

            print("✅ Модель успешно загружена")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"❌ Ошибка при загрузке модели: {e}")
            return False

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str = "class",
    ) -> Tuple[float, str]:
        """
        Обучение модели на DataFrame с теми же колонками, что и в train.csv.

        Ожидается, что в df есть:
        - столбец target_column ('class'),
        - признак 'id' (будет проигнорирован),
        - остальные признаки, как в исходном датасете.
        """

        if target_column not in df.columns:
            raise ValueError(f"В датафрейме нет столбца таргета '{target_column}'")

        # отделяем таргет
        y_raw = df[target_column]

        # убираем id и таргет из признаков
        X = df.drop(columns=[target_column, "id"], errors="ignore")

        # запомним список признаков (в таком порядке модель их и видит)
        feature_names = list(X.columns)

        # кодируем категориальные признаки
        label_encoders: Dict[str, LabelEncoder] = {}
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # кодируем таргет: e=0, p=1
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))

        # train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        now_iso = datetime.now().isoformat()

        # сохраняем в pkl всё, что нужно для инференса
        model_data: Dict[str, Any] = {
            "model": model,
            "label_encoders": label_encoders,
            "features": feature_names,
            "model_trained_date": now_iso,
        }

        joblib.dump(model_data, self.model_path)
        print(f"💾 Модель сохранена в {self.model_path}")
        print(f"📊 Точность на тесте: {accuracy:.4f}")

        # обновляем состояние объекта
        self.model = model
        self.label_encoders = label_encoders
        self.features = feature_names
        self.model_trained_date = now_iso
        self.model_loaded_date = now_iso

        return accuracy, now_iso

    def get_model_info(self) -> dict:
        """Информация о модели (для эндпоинта /status)."""
        return {
            "is_loaded": self.model is not None,
            "model_loaded_date": self.model_loaded_date,
            "model_trained_date": self.model_trained_date,
            "features": self.features,
            "model_path": self.model_path,
        }
        

model_config = ModelConfig()
