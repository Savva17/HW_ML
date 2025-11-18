import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


# -------------------------------
# 1. Загружаем датасет
# -------------------------------
df = pd.read_csv('data/train.csv')
df_small = df.sample(n=500, random_state=42)
print(f"📊 Загружено {len(df_small)} записей (случайная выборка)")

df_small.to_csv('data/train_sample.csv', index=False)

X = df_small.drop('class', axis=1)
y = df_small['class']

# -------------------------------
# 2. Препроцессинг
# -------------------------------
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

y = LabelEncoder().fit_transform(y)  # e=0, p=1

# -------------------------------
# 3. Тренируем модель
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print(f"📊 Accuracy: {model.score(X_test, y_test):.4f}")

# -------------------------------
# 4. Сохраняем модель и энкодеры
# -------------------------------
model_data = {
    'model': model,
    'label_encoders': label_encoders
}

joblib.dump(model_data, "mushroom_model.pkl")
print("💾 Model saved to mushroom_model.pkl")
