import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


# -------------------------------
# 1. Загружаем датасет
# -------------------------------
df = pd.read_csv('data/train.csv')
df_small = df.sample(n=1000, random_state=42)
print(f"📊 Загружено {len(df_small)} записей (случайная выборка)")

df_small.to_csv('data/train_sample.csv', index=False)

X = df_small.drop('class', axis=1)
y_raw = df_small['class']

# Преобразуем таргет в 0/1: 0 = съедобный (e), 1 = ядовитый (p)
y = (y_raw == "p").astype(int)

# -------------------------------
# 2. Препроцессинг
# -------------------------------
categorical_features = list(X.columns)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------
# 3. Тренируем модель
# -------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print(f"📊 Accuracy: {model.score(X_test, y_test):.4f}")

# -------------------------------
# 4. Сохраняем модель и энкодеры
# -------------------------------
joblib.dump(model, "mushroom_model.pkl")
print("💾 Model saved to mushroom_model.pkl")
