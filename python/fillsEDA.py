import os
import clickhouse_connect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------------------
# 0. Config -- adjust to your environment
# ---------------------------------------------------------------------------
CLICKHOUSE_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "default",
    "password": "",
    "database": "hyperliquid",
}
 
FILLS_TABLE = "hyperliquid.fills"
SAMPLE_META_PATH = "output/addresses_sample.parquet"
OUTPUT_DIR = "output/edaOutputs"
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def loadFills(client):
    # ReplacingMergeTree only merges duplicates in the background, so a plain
    # SELECT can double-count rows for the same tid if a merge hasn't run
    # yet. FINAL forces dedup at query time -- slower, but correct, and this
    # is a one-off EDA pull on a 320-address sample so the cost is fine.
    query = f"""
        SELECT
            address,
            coin,
            side,
            px,
            sz,
            time,
            dir,
            start_position AS startPosition,
            closed_pnl AS closedPnl,
            fee,
            fee_token AS feeToken,
            crossed,
            tid
        FROM {FILLS_TABLE} FINAL
    """
    return client.query_df(query)
 
 
client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
fillsDf = loadFills(client)
 
# Belt-and-suspenders dedup: FINAL can still miss very recent inserts that
# haven't been picked up by a merge cycle yet, so drop on tid as well.
fillsDf = fillsDf.drop_duplicates(subset="tid", keep="last")
 
metaDf = pd.read_parquet(SAMPLE_META_PATH)
metaDf = metaDf.rename(columns={
    "ethAddress": "address",
    "allTime_vlm": "allTimeVlm",
    "allTime_roi": "allTimeRoi",
    "allTime_pnl": "allTimePnl",
})
 
fillsDf["time"] = pd.to_datetime(fillsDf["time"])
mergedDf = fillsDf.merge(
    metaDf[["address", "statusTier", "accountValue", "allTimeVlm", "allTimeRoi", "allTimePnl"]],
    on="address",
    how="left",
)
 
print(f"Total fills loaded (post-dedup): {len(fillsDf):,}")
print(f"Unique addresses in fills: {fillsDf['address'].nunique()} / {metaDf['address'].nunique()} sampled")
 
# ---------------------------------------------------------------------------
# 2. Coverage check -- did every sampled address actually get fills data?
# ---------------------------------------------------------------------------
fillsPerAddress = fillsDf.groupby("address").size().rename("fillCount")
coverageDf = metaDf.set_index("address").join(fillsPerAddress, how="left")
coverageDf["fillCount"] = coverageDf["fillCount"].fillna(0).astype(int)
 
missingAddresses = coverageDf[coverageDf["fillCount"] == 0]
print(f"\nAddresses with ZERO fills returned: {len(missingAddresses)}")
if len(missingAddresses) > 0:
    print(missingAddresses.groupby("statusTier").size())
 
print("\nFill count distribution by statusTier:")
print(coverageDf.groupby("statusTier")["fillCount"].describe())
 
# ---------------------------------------------------------------------------
# 3. Time span & recency -- key for the churn narrative
# ---------------------------------------------------------------------------
timeSpanDf = mergedDf.groupby("address").agg(
    firstFill=("time", "min"),
    lastFill=("time", "max"),
    fillCount=("time", "size"),
)
timeSpanDf["activeSpanDays"] = (timeSpanDf["lastFill"] - timeSpanDf["firstFill"]).dt.days
now = pd.Timestamp.utcnow().tz_localize(None)
timeSpanDf["daysSinceLastFill"] = (now - timeSpanDf["lastFill"]).dt.days
 
timeSpanDf = timeSpanDf.join(metaDf.set_index("address")["statusTier"])
 
print("\nDays since last fill, by statusTier (this is the core churn signal):")
print(timeSpanDf.groupby("statusTier")["daysSinceLastFill"].describe())
 
fig, ax = plt.subplots(figsize=(9, 5))
for tier, group in timeSpanDf.groupby("statusTier"):
    ax.hist(group["daysSinceLastFill"].clip(upper=180), bins=30, alpha=0.5, label=tier)
ax.set_xlabel("Days since last fill (clipped at 180)")
ax.set_ylabel("Address count")
ax.set_title("Recency distribution by statusTier")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/recency_by_tier.png", dpi=150)
plt.close(fig)
 
# ---------------------------------------------------------------------------
# 4. Trade structure -- notional size, side, direction mix
# ---------------------------------------------------------------------------
mergedDf["notional"] = mergedDf["px"] * mergedDf["sz"]
 
print("\nNotional per fill, by statusTier:")
print(mergedDf.groupby("statusTier")["notional"].describe())
 
sideBreakdown = mergedDf.groupby(["statusTier", "side"]).size().unstack(fill_value=0)
sideBreakdownPct = sideBreakdown.div(sideBreakdown.sum(axis=1), axis=0)
print("\nBuy/sell side mix by statusTier (share of fills):")
print(sideBreakdownPct.round(3))
 
dirBreakdown = mergedDf.groupby(["statusTier", "dir"]).size().unstack(fill_value=0)
dirBreakdownPct = dirBreakdown.div(dirBreakdown.sum(axis=1), axis=0)
print("\nTrade direction mix by statusTier (open/close, long/short):")
print(dirBreakdownPct.round(3))
 
# ---------------------------------------------------------------------------
# 5. Realized PnL structure -- win rate & payoff shape per tier
# ---------------------------------------------------------------------------
closedTradesDf = mergedDf[mergedDf["closedPnl"] != 0].copy()
closedTradesDf["isWin"] = closedTradesDf["closedPnl"] > 0
 
winRateByTier = closedTradesDf.groupby("statusTier")["isWin"].mean()
avgWinByTier = closedTradesDf[closedTradesDf["isWin"]].groupby("statusTier")["closedPnl"].mean()
avgLossByTier = closedTradesDf[~closedTradesDf["isWin"]].groupby("statusTier")["closedPnl"].mean()
 
payoffSummary = pd.DataFrame({
    "winRate": winRateByTier,
    "avgWin": avgWinByTier,
    "avgLoss": avgLossByTier,
})
print("\nWin rate & average win/loss, by statusTier:")
print(payoffSummary.round(3))
 
# ---------------------------------------------------------------------------
# 6. Fee burden -- can matter a lot for the activeLongTail "struggling" story
# ---------------------------------------------------------------------------
feeSummary = mergedDf.groupby("statusTier").agg(
    totalFee=("fee", "sum"),
    totalNotional=("notional", "sum"),
)
feeSummary["feeAsPctOfNotional"] = feeSummary["totalFee"] / feeSummary["totalNotional"]
print("\nFee burden by statusTier:")
print(feeSummary.round(6))
 
# ---------------------------------------------------------------------------
# 7. Trading frequency over the observed window -- flag possible pre-churn spikes
# ---------------------------------------------------------------------------
mergedDf["fillDate"] = mergedDf["time"].dt.date
dailyFreqDf = (
    mergedDf.groupby(["address", "fillDate"]).size().rename("dailyFills").reset_index()
)
dailyFreqDf = dailyFreqDf.merge(metaDf[["address", "statusTier"]], on="address", how="left")
 
avgDailyFreqByTier = dailyFreqDf.groupby("statusTier")["dailyFills"].mean()
print("\nAverage daily fill count on active days, by statusTier:")
print(avgDailyFreqByTier.round(2))
 
# ---------------------------------------------------------------------------
# 8. Save merged working table for downstream Layer 1 analyses
# ---------------------------------------------------------------------------
mergedDf.to_parquet(f"{OUTPUT_DIR}/fills_merged_with_tier.parquet", index=False)
timeSpanDf.to_parquet(f"{OUTPUT_DIR}/address_time_span.parquet")
 
print(f"\nDone. Plots and merged tables saved to ./{OUTPUT_DIR}/")