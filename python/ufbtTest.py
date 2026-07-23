# To check sample fills

import json
import requests
from datetime import datetime


API_URL = "https://api.hyperliquid.xyz/info"


def get_user_fills(address, limit=10):
    payload = {
        "type": "userFillsByTime",
        "user": address,
        "startTime": 0
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    fills = response.json()

    return fills[:limit]


def format_time(timestamp):
    return datetime.fromtimestamp(
        timestamp / 1000
    ).isoformat()


def main():
    address = input(
        "Input Hyperliquid address: "
    ).strip()

    fills = get_user_fills(
        address,
        limit=10
    )

    print(
        f"\nFound {len(fills)} fills\n"
    )

    for i, fill in enumerate(fills):
        print("=" * 80)

        print(f"#{i + 1}")

        for key, value in fill.items():
            if key == "time":
                print(
                    f"{key}: {value} ({format_time(value)})"
                )
            else:
                print(
                    f"{key}: {value}"
                )

    with open(
        "output/sample_fills.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            fills,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nSaved to sample_fills.json"
    )


if __name__ == "__main__":
    main()