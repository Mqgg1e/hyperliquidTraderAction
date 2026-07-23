import json
import time
import random
from datetime import datetime, timezone

import requests
import clickhouse_connect
import pandas as pd


API_URL = "https://api.hyperliquid.xyz/info"
ADDRESSES_FILE = "output/addresses_sample.json"
MAX_FILLS_PER_ADDRESS = 10000      
PAGE_LIMIT = 2000                  
SLEEP_BETWEEN_REQUESTS =2.6
MAX_RETRIES = 5
BATCH_INSERT_SIZE = 5000           

CH_HOST = "localhost"
CH_PORT = 8123
PARQUET_FILE = "output/addresses_sample.parquet"


def loadAddresses(path):
    with open(path, "r") as f:
        return json.load(f)


def postWithRetry(session, payload):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(API_URL, json=payload, timeout=15)
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.random()
            print(f"  Network error {e}, retry after {wait:.1f}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            wait = (2 ** attempt) * 5 + random.random()
            print(f"  429 limit, retry after{wait:.1f}s")
            time.sleep(wait)
            continue

        print(f" code {resp.status_code}: {resp.text[:200]} received")
        time.sleep(3)

    raise RuntimeError(f"Failed for {MAX_RETRIES} times, payload={payload}")


def fetchFillsForAddress(session, address):
    allFills = []
    startTime = 0

    while True:
        payload = {
            "type": "userFillsByTime",
            "user": address,
            "startTime": startTime,
            "aggregateByTime": False,
        }
        fills = postWithRetry(session, payload)

        if not fills:
            break

        allFills.extend(fills)

        if len(fills) < PAGE_LIMIT:
            break
        if len(allFills) >= MAX_FILLS_PER_ADDRESS:
            break

        startTime = fills[-1]["time"] + 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return allFills[:MAX_FILLS_PER_ADDRESS]


def toRow(address, fill):
    return (
        address,
        fill["coin"],
        float(fill["px"]),
        float(fill["sz"]),
        fill["side"],
        datetime.fromtimestamp(fill["time"] / 1000, tz=timezone.utc),
        float(fill["startPosition"]),
        fill["dir"],
        float(fill["closedPnl"]),
        fill["hash"],
        int(fill["oid"]),
        bool(fill["crossed"]),
        float(fill["fee"]),
        int(fill["tid"]),
        fill.get("cloid"),
        fill["feeToken"],
        fill.get("twapId"),
    )


COLUMNS = [
    "address", "coin", "px", "sz", "side", "time", "start_position",
    "dir", "closed_pnl", "hash", "oid", "crossed", "fee", "tid",
    "cloid", "fee_token", "twap_id",
]


def insertFills(client, address, fills):
    if not fills:
        return 0

    rows = [toRow(address, f) for f in fills]

    for i in range(0, len(rows), BATCH_INSERT_SIZE):
        chunk = rows[i:i + BATCH_INSERT_SIZE]
        client.insert("hyperliquid.fills", chunk, column_names=COLUMNS)

    return len(rows)


def insert_addresses_from_parquet(client, parquet_path):
    """Read a parquet file and insert rows into hyperliquid.addresses.

    Expects columns: ethAddress, statusTier, accountValue, allTime_vlm,
    allTime_roi, allTime_pnl (per user's DataFrame schema). Maps them to
    the ClickHouse table columns and inserts in batches.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Failed to read parquet {parquet_path}: {e}")
        return 0

    # Normalize/rename columns to DB schema
    rename_map = {
        "ethAddress": "address",
        "statusTier": "tier",
        "accountValue": "account_value",
        "allTime_pnl": "pnl",
        "allTime_roi": "roi",
        "allTime_vlm": "volume",
    }
    df = df.rename(columns=rename_map)

    cols = ["address", "tier", "account_value", "pnl", "roi", "volume"]

    # Ensure the expected columns exist
    for c in cols:
        if c not in df.columns:
            print(f"Missing column in parquet: {c}")
            return 0

    # Fill NaNs for numeric columns to avoid insert issues
    df["account_value"] = df["account_value"].fillna(0).astype(float)
    df["pnl"] = df["pnl"].fillna(0).astype(float)
    df["roi"] = df["roi"].fillna(0).astype(float)
    df["volume"] = df["volume"].fillna(0).astype(float)

    rows = [tuple(row[c] for c in cols) for _, row in df.iterrows()]

    n = 0
    for i in range(0, len(rows), BATCH_INSERT_SIZE):
        chunk = rows[i:i + BATCH_INSERT_SIZE]
        client.insert("hyperliquid.addresses", chunk, column_names=cols)
        n += len(chunk)

    return n


def alreadyDone(client, address):
    result = client.query(
        "SELECT status FROM hyperliquid.fetch_log WHERE address = {addr:String} ORDER BY updated_at DESC LIMIT 1",
        parameters={"addr": address},
    )
    if result.result_rows:
        return result.result_rows[0][0] == "success"
    return False


def logStatus(client, address, status, nFills):
    client.insert(
        "hyperliquid.fetch_log",
        [(address, status, datetime.now(timezone.utc), nFills, datetime.now(timezone.utc))],
        column_names=["address", "status", "last_fetched_time", "n_fills", "updated_at"],
    )


def main():
    addresses = loadAddresses(ADDRESSES_FILE)
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)
    session = requests.Session()

    print(f"Up to {len(addresses)} addresses to be handled.")

    for idx, address in enumerate(addresses, 1):
        if alreadyDone(client, address):
            print(f"[{idx}/{len(addresses)}] {address} has been processed. Continue.")
            continue

        print(f"[{idx}/{len(addresses)}] taking {address} ...")
        try:
            fills = fetchFillsForAddress(session, address)
            n = insertFills(client, address, fills)
            logStatus(client, address, "success", n)
            print(f"  Done, {n} lines have been written")
        except Exception as e:
            print(f"  Failed: {e}")
            logStatus(client, address, "failed", 0)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # After processing fills, attempt to load and insert addresses parquet
    try:
        print(f"Inserting addresses from parquet: {PARQUET_FILE} ...")
        n_addr = insert_addresses_from_parquet(client, PARQUET_FILE)
        if n_addr:
            print(f"  Inserted {n_addr} addresses into hyperliquid.addresses")
        else:
            print("  No addresses inserted (file missing or empty).")
    except Exception as e:
        print(f"  Failed inserting addresses parquet: {e}")

    print("All processes finished.")


if __name__ == "__main__":
    main()