from urllib import request

from ..app.config import BASE_URL


def main() -> None:
    """
    Тестирует эндпоинт POST /fit
    Переобучает модель на train_sample.csv
    """
    url = f"{BASE_URL}/fit"

    print(f"POST {url}")
    req = request.Request(url, data=b"", method="POST")
    req.add_header("Accept", "application/json")

    with request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print("Status:", resp.status)
        print("Response:", body)


if __name__ == "__main__":
    main()
