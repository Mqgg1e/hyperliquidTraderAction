import pandas as pd
import numpy as np
import os

INPUT_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"  # adjust to your actual saved filename
mergedDf = pd.read_parquet(INPUT_PATH)
mergedDf["time"] = pd.to_datetime(mergedDf["time"])

OUTPUT_DIR = "output/churnMechanism"
 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Full list of dir categories -- is Liquidation actually present?
# ---------------------------------------------------------------------------
print("All distinct dir values:")
print(mergedDf["dir"].value_counts())

liquidationTagged = mergedDf[mergedDf["dir"].str.contains("Liquidat", case=False, na=False)]
print(f"\nFills tagged as liquidation-related: {len(liquidationTagged):,}")
if len(liquidationTagged) > 0:
    print(liquidationTagged.groupby("statusTier").size())

# ---------------------------------------------------------------------------
# 2. Loss concentration -- how much of total loss comes from the single
#    worst trade per address, by tier?
# ---------------------------------------------------------------------------
lossesDf = mergedDf[mergedDf["closedPnl"] < 0]

lossConcentration = lossesDf.groupby(["statusTier", "address"]).agg(
    totalLoss=("closedPnl", "sum"),
    worstSingleLoss=("closedPnl", "min"),
).reset_index()
lossConcentration["worstLossShare"] = lossConcentration["worstSingleLoss"] / lossConcentration["totalLoss"]

print("\nShare of total loss coming from the single worst trade, by tier:")
print(lossConcentration.groupby("statusTier")["worstLossShare"].describe())

# ---------------------------------------------------------------------------
# 3. Position size escalation -- compare notional in an address's first
#    20% of fills vs their last 20% of fills before going quiet
# ---------------------------------------------------------------------------
def escalationRatio(group):
    group = group.sort_values("time")
    n = len(group)
    if n < 10:
        return np.nan
    cutoff = max(1, n // 5)
    earlyMean = group["notional"].iloc[:cutoff].mean()
    lateMean = group["notional"].iloc[-cutoff:].mean()
    if earlyMean == 0:
        return np.nan
    return lateMean / earlyMean

escalationByAddress = mergedDf.groupby(["statusTier", "address"]).apply(escalationRatio)
escalationByAddress = escalationByAddress.rename("escalationRatio").reset_index()

print("\nPosition size escalation ratio (last 20% notional / first 20% notional), by tier:")
print(escalationByAddress.groupby("statusTier")["escalationRatio"].describe())

# ---------------------------------------------------------------------------
# 4. Trading frequency in the final week of activity vs overall average
#    -- flags a "going out in a blaze" pattern right before churn
# ---------------------------------------------------------------------------
def finalWeekIntensity(group):
    group = group.sort_values("time")
    lastTime = group["time"].max()
    finalWeekCount = (group["time"] >= lastTime - pd.Timedelta(days=7)).sum()
    totalDays = max(1, (group["time"].max() - group["time"].min()).days)
    overallDaily = len(group) / totalDays
    return pd.Series({
        "finalWeekFillCount": finalWeekCount,
        "overallDailyRate": overallDaily,
        "finalWeekDailyRate": finalWeekCount / 7,
    })

intensityDf = mergedDf.groupby(["statusTier", "address"]).apply(finalWeekIntensity).reset_index()
intensityDf["intensityRatio"] = intensityDf["finalWeekDailyRate"] / intensityDf["overallDailyRate"].replace(0, np.nan)

print("\nFinal week trading intensity vs overall average, by tier (>1 = sped up before going quiet):")
print(intensityDf.groupby("statusTier")["intensityRatio"].describe())

# ---------------------------------------------------------------------------
# 5. Save for later use
# ---------------------------------------------------------------------------
lossConcentration.to_parquet("output/churnMechanism/lossConcentrationByAddress.parquet", index=False)
escalationByAddress.to_parquet("output/churnMechanism/escalationByAddress.parquet", index=False)
intensityDf.to_parquet("output/churnMechanism/finalWeekIntensityByAddress.parquet", index=False)

print("\nDone.")