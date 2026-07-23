import pandas as pd
import numpy as np
 
mergedDf = pd.read_parquet("output/edaOutputs/fills_merged_with_tier.parquet")
mergedDf["time"] = pd.to_datetime(mergedDf["time"])
 
closedDf = mergedDf[mergedDf["closedPnl"] != 0].copy()
closedDf["isWin"] = closedDf["closedPnl"] > 0
 
# ---------------------------------------------------------------------------
# 1. Split each account's closed-trade history into quintiles (by trade
#    order, not calendar time) and compute win rate + mean PnL per quintile.
#    This directly tests "declining edge over the account's lifetime".
# ---------------------------------------------------------------------------
def quintileStats(group):
    group = group.sort_values("time").reset_index(drop=True)
    n = len(group)
    if n < 20:
        return None
    group["quintile"] = pd.qcut(group.index, 5, labels=False, duplicates="drop")
    stats = group.groupby("quintile").agg(
        winRate=("isWin", "mean"),
        meanPnl=("closedPnl", "mean"),
        n=("closedPnl", "size"),
    )
    return stats
 
 
records = []
for (tier, address), group in closedDf.groupby(["statusTier", "address"]):
    stats = quintileStats(group)
    if stats is None:
        continue
    for quintile, row in stats.iterrows():
        records.append({
            "statusTier": tier,
            "address": address,
            "quintile": quintile,
            "winRate": row["winRate"],
            "meanPnl": row["meanPnl"],
        })
 
quintileDf = pd.DataFrame(records)
 
print("Win rate by lifetime quintile (0=earliest trades, 4=latest trades), by tier:")
print(quintileDf.groupby(["statusTier", "quintile"])["winRate"].mean().unstack().round(3))
 
print("\nMean PnL per closed trade by lifetime quintile, by tier:")
print(quintileDf.groupby(["statusTier", "quintile"])["meanPnl"].mean().unstack().round(2))
 
# ---------------------------------------------------------------------------
# 2. Per-address linear trend: is win rate trending down across quintiles?
#    Slope of winRate ~ quintile, one number per address.
# ---------------------------------------------------------------------------
def winRateSlope(subDf):
    if subDf["quintile"].nunique() < 3:
        return np.nan
    return np.polyfit(subDf["quintile"], subDf["winRate"], 1)[0]
 
slopeDf = quintileDf.groupby(["statusTier", "address"]).apply(winRateSlope).rename("winRateSlope").reset_index()
 
print("\nPer-address win-rate slope across lifetime quintiles (negative = declining edge), by tier:")
print(slopeDf.groupby("statusTier")["winRateSlope"].describe())
 
shareDecliningByTier = slopeDf.groupby("statusTier")["winRateSlope"].apply(lambda s: (s < 0).mean())
print("\nShare of addresses with a NEGATIVE win-rate slope (declining edge), by tier:")
print(shareDecliningByTier.round(3))
 
# ---------------------------------------------------------------------------
# 3. Does tapering (lower late-period activity) correlate with a rising
#    loss rate, rather than reflecting one big loss? Compare win rate in
#    an address's final quintile vs its first quintile.
# ---------------------------------------------------------------------------
firstLastDf = quintileDf[quintileDf["quintile"].isin([0, 4])].pivot(
    index=["statusTier", "address"], columns="quintile", values="winRate"
)
firstLastDf.columns = ["winRateFirst", "winRateLast"]
firstLastDf["winRateChange"] = firstLastDf["winRateLast"] - firstLastDf["winRateFirst"]
firstLastDf = firstLastDf.reset_index()
 
print("\nWin rate change from first to last lifetime quintile, by tier:")
print(firstLastDf.groupby("statusTier")["winRateChange"].describe())
 
# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
import os
os.makedirs("output/churnMechanism", exist_ok=True)
quintileDf.to_parquet("output/churnMechanism/pnlQuintileByAddress.parquet", index=False)
slopeDf.to_parquet("output/churnMechanism/winRateSlopeByAddress.parquet", index=False)
 
print("\nDone.")