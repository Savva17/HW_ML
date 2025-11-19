import json
from urllib import request

from ..app.config import BASE_URL


def main() -> None:
    """
    Тестирует эндпоинт POST /predict_batch
    """
    url = f"{BASE_URL}/predict_batch"
    payload = {
        "mushrooms": [
            {"cap_color": "n", "habitat": "u"},
            {"cap_color": "y", "habitat": "g"},
            {"cap_color": "w", "habitat": "d"},
        ]
    }

    data_bytes = json.dumps(payload).encode("utf-8")

    print(f"POST {url}")
    print("Payload:", json.dumps(payload, indent=2, ensure_ascii=False))

    req = request.Request(url, data=data_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print("Status:", resp.status)
        print("Response:", body)


if __name__ == "__main__":
    main()
