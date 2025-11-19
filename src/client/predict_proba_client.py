import json
from urllib import request, parse

from ..app.config import BASE_URL


def main() -> None:
    """
    Тестирует эндпоинт GET /predict_proba
    """
    params = {
        "cap_color": "y",
        "habitat": "g",
    }
    query = parse.urlencode(params)
    url = f"{BASE_URL}/predict_proba?{query}"

    print(f"GET {url}")
    req = request.Request(url, method="GET")
    req.add_header("Accept", "application/json")

    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print("Status:", resp.status)
        print("Response:", body)


if __name__ == "__main__":
    main()
