# 🍄 Precision_Mushrooms 

[![Python](https://img.shields.io/badge/-Python-464646?style=flat-square&logo=Python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/-FastAPI-464646?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/-Pydantic-464646?style=flat&logo=pydantic)](https://docs.pydantic.dev/)
[![docker](https://img.shields.io/badge/-Docker-464646?style=flat-square&logo=docker)](https://www.docker.com/)

## Описание
Проект Precisio_Mushrooms — это FastAPI-сервис, который определяет, является ли гриб ядовитым, основываясь на его характеристиках.

### Как запустить проект:
Клонировать репозиторий и перейти в него в командной строке:

```
git clone https://github.com/Savva17/HW_ML.git
```

```
cd HW_ML
```

Cоздать и активировать виртуальное окружение:

```
python3 -m venv venv
```

* Если у вас Linux/macOS

    ```
    source venv/bin/activate
    ```

* Если у вас windows

    ```
    source venv/scripts/activate
    ```

Установить зависимости из файла requirements.txt:

```
python3 -m pip install --upgrade pip
```

```
pip install -r requirements.txt
```

## Запуск проекта

```
python3 -m src.app.main
```
Документация Swagger по адресу:
```
http://127.0.0.1:8002/docs
```

## Запуск проекта через Dockerfile
Сборка Docker-образа
```
docker build -t mushroom-api .
```
Запуск контейнера
```
docker run -p 8002:8002 mushroom-api
```
Документация Swagger по адресу:
```
http://127.0.0.1:8002/docs
```

## Пример запросов API
- (**GET**): Предсказание одного гриба:<br />
```/predict?cap_color=n&habitat=u```
- (**GET**): Предсказание вероятности:<br />
```/predict_proba?cap_color=y&habitat=g```
- (**POST**): Предсказание для списка грибов:<br />
```
POST /predict_batch
{
  "mushrooms": [
    { "cap_color": "n", "habitat": "u" },
    { "cap_color": "y", "habitat": "g" }
  ]
}
```
- (**POST**): Вероятности для списка грибов:<br />
```
POST /predict_proba_batch
{
  "mushrooms": [
    { "cap_color": "n", "habitat": "u" },
    { "cap_color": "y", "habitat": "g" }
  ]
}
```
- (**GET**): Статус модели:<br />
```/status```
- (**GET**): Переобучение модели:<br />
```/fit```

## Работа клиента
### Для их запуска убедитесь, что сервер работает!!!
Проверить /predict
```
python -m src.client.predict_client
```
Проверить /predict_proba
```
python -m src.client.predict_proba_client
```
Проверить /predict_batch
```
python -m src.client.predict_batch_client
```
Проверить /predict_proba_batch
```
python -m src.client.predict_proba_batch_client
```
Проверить /status
```
python -m src.client.status_client
```
Проверить /fit
```
python -m src.client.fit_client
```


Автор проекта: Морозов Савва

Профиль автора на GitHub:
- **GitHub**: [Профиль Савва Морозов](https://github.com/Savva17)

