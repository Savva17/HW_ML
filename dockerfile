# версия питона
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY . .
EXPOSE 8002

# При старте контейнера запустить сервер разработки.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
