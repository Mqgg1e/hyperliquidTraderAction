
import os
 
import numpy as np
import pandas as pd
 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
 
INPUT_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
OUTPUT_DIR = "output/churnMechanism"
MAX_MONTHS_BEFORE_LAST_FILL = 3
MIN_RELIABLE_N = 10
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
fillsDf = pd.read_parquet(INPUT_PATH)
fillsDf["time"] = pd.to_datetime(fillsDf["time"])
fillsDf["fillMonth"] = fillsDf["time"].dt.to_period("M")
 
# lastFill is computed from ALL fills (not just closed trades), consistent
# with the definition already used in layer1BehaviorAnalysis.py.
lastFillByAddress = (
    fillsDf.groupby(["statusTier", "address"])["fillMonth"].max().rename("lastFillMonth").reset_index()
)
 
closedDf = fillsDf[fillsDf["closedPnl"] != 0].copy()
closedDf = closedDf.merge(lastFillByAddress, on=["statusTier", "address"], how="left")
closedDf["monthsBeforeLastFill"] = (
    (closedDf["lastFillMonth"].dt.year - closedDf["fillMonth"].dt.year) * 12
    + (closedDf["lastFillMonth"].dt.month - closedDf["fillMonth"].dt.month)
)
 
closedDf = closedDf[closedDf["monthsBeforeLastFill"].between(0, MAX_MONTHS_BEFORE_LAST_FILL)]
 
# ---------------------------------------------------------------------------
# 1. Address x bucket level: net PnL and win rate for that calendar month.
# ---------------------------------------------------------------------------
addressBucket = closedDf.groupby(["statusTier", "address", "monthsBeforeLastFill"]).agg(
    netPnl=("closedPnl", "sum"),
    tradeCount=("closedPnl", "size"),
    winRate=("closedPnl", lambda s: (s > 0).mean()),
).reset_index()
addressBucket["netProfitable"] = addressBucket["netPnl"] > 0
 
print("Address x monthsBeforeLastFill rows:", len(addressBucket))
 
# ---------------------------------------------------------------------------
# 2. Tier x bucket summary: share of accounts net-profitable that month,
#    plus mean/median netPnl (context only - flag if mean/median ratio > 5x).
# ---------------------------------------------------------------------------
def summarizeBucket(group):
    n = len(group)
    medianPnl = group["netPnl"].median()
    meanPnl = group["netPnl"].mean()
    ratio = np.nan if medianPnl == 0 else abs(meanPnl / medianPnl)
    return pd.Series({
        "n": n,
        "shareNetProfitable": group["netProfitable"].mean(),
        "meanWinRate": group["winRate"].mean(),
        "medianNetPnl": medianPnl,
        "meanNetPnl": meanPnl,
        "meanMedianRatio": ratio,
        "reliableN": n >= MIN_RELIABLE_N,
    })
 
summaryDf = (
    addressBucket.groupby(["statusTier", "monthsBeforeLastFill"])
    .apply(summarizeBucket, include_groups=False)
    .reset_index()
)
 
print("\nShare of accounts net-profitable by statusTier x monthsBeforeLastFill "
      "(0 = the month containing the address's last fill):")
pivotShare = summaryDf.pivot(index="statusTier", columns="monthsBeforeLastFill", values="shareNetProfitable")
print(pivotShare.round(3))
 
print("\nSample size (n addresses) per cell:")
pivotN = summaryDf.pivot(index="statusTier", columns="monthsBeforeLastFill", values="n")
print(pivotN)
 
print("\nCells with n < {}: flag as unreliable, do not use for conclusions.".format(MIN_RELIABLE_N))
print(summaryDf[~summaryDf["reliableN"]][["statusTier", "monthsBeforeLastFill", "n"]])
 
# ---------------------------------------------------------------------------
# 3. Trend check per tier: is shareNetProfitable declining as
#    monthsBeforeLastFill -> 0 (i.e. approaching the last observed fill)?
#    Slope of shareNetProfitable ~ monthsBeforeLastFill: POSITIVE slope means
#    edge was higher further from the last fill, i.e. edge declined toward
#    disengagement.
# ---------------------------------------------------------------------------
trendRows = []
for tier, group in summaryDf[summaryDf["reliableN"]].groupby("statusTier"):
    if group["monthsBeforeLastFill"].nunique() < 3:
        trendRows.append({"statusTier": tier, "slope": np.nan, "reliableBuckets": group["monthsBeforeLastFill"].nunique()})
        continue
    slope = np.polyfit(group["monthsBeforeLastFill"], group["shareNetProfitable"], 1)[0]
    trendRows.append({"statusTier": tier, "slope": slope, "reliableBuckets": group["monthsBeforeLastFill"].nunique()})
trendDf = pd.DataFrame(trendRows)
print("\nSlope of shareNetProfitable ~ monthsBeforeLastFill, by tier "
      "(positive slope = edge was better further from the last fill = edge declined toward disengagement):")
print(trendDf.round(4))
 
# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
addressBucket.to_parquet(f"{OUTPUT_DIR}/netPnlByAddressMonthBucket.parquet", index=False)
summaryDf.to_parquet(f"{OUTPUT_DIR}/shareNetProfitableByTierBucket.parquet", index=False)
 
print(f"\nSaved to {OUTPUT_DIR}/")
 