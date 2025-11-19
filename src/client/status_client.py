from urllib import request

from ..app.config import BASE_URL


def main() -> None:
    """
    Тестирует эндпоинт GET /status
    """
    url = f"{BASE_URL}/status"

    print(f"GET {url}")
    req = request.Request(url, method="GET")
    req.add_header("Accept", "application/json")

    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print("Status:", resp.status)
        print("Response:", body)


if __name__ == "__main__":
    main()
