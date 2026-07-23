### Create tables

-- SHOW DATABASES;

-- CREATE DATABASE IF NOT EXISTS hyperliquid;

-- ReplacingMergeTree (address, time, tid), in case rewrite when continue after break
/*
CREATE TABLE hyperliquid.fills
(
    address      String,                        
    coin         LowCardinality(String),
    px           Float64,
    sz           Float64,
    side         LowCardinality(String),         -- "B" / "A"
    time         DateTime64(3),                  -- Originated from ms to DateTime64(3)
    start_position Float64,
    dir          LowCardinality(String),         -- "Open Long" / "Close Short"
    closed_pnl   Float64,
    hash         String,
    oid          Int64,
    crossed      Bool,
    fee          Float64,
    tid          Int64,                          -- Order filled id.
    cloid        Nullable(String),
    fee_token    LowCardinality(String),
    twap_id      Nullable(Int64),
    inserted_at  DateTime DEFAULT now()   
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(time)
ORDER BY (address, time, tid);
*/

/*
-- Address name list and snapshot information
CREATE TABLE hyperliquid.addresses
(
    address       String,
    tier          LowCardinality(String),        -- 'whale' / 'active' / 'longtail' tier label
    account_value Float64,
    pnl           Float64,
    roi           Float64,
    volume        Float64,
    snapshot_time DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY address;
*/


/*
-- fetch status
CREATE TABLE hyperliquid.fetch_log
(
    address       String,
    status        LowCardinality(String),        -- 'pending' / 'success' / 'failed'
    last_fetched_time DateTime64(3),              -- Last order time fetched.
    n_fills        UInt32,
    updated_at     DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY address;
*/

SHOW tables from hyperliquid;



### Data verification
/*SELECT
    fl.address,
    fl.n_fills AS logged,
    count(*) AS actual
FROM hyperliquid.fetch_log fl
LEFT JOIN hyperliquid.fills f ON f.address = fl.address
GROUP BY fl.address, fl.n_fills
HAVING logged != actual
LIMIT 20;
*/

/*
SELECT address FROM hyperliquid.fetch_log
WHERE status = 'success' AND n_fills = 0;
*/

SELECT
    address,
    min(time) AS earliest,
    max(time) AS latest,
    count(*) AS n
FROM hyperliquid.fills
GROUP BY address
ORDER BY n ASC
LIMIT 20;
