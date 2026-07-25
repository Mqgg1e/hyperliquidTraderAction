import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

INPUT_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
OUTPUT_DIR = "output/layer1Behavior"

os.makedirs(OUTPUT_DIR, exist_ok=True)

fillsDf = pd.read_parquet(INPUT_PATH)
fillsDf["time"] = pd.to_datetime(fillsDf["time"])
fillsDf["fillDate"] = fillsDf["time"].dt.date
fillsDf["fillMonth"] = fillsDf["time"].dt.to_period("M").astype(str)

closedDf = fillsDf[fillsDf["closedPnl"] != 0].copy()
closedDf["isWin"] = closedDf["closedPnl"] > 0

# ---------------------------------------------------------------------------
# 1. Address-level feature table for Layer 1 behavior analysis
# ---------------------------------------------------------------------------
addressFeatures = fillsDf.groupby(["statusTier", "address"]).agg(
    firstFill=("time", "min"),
    lastFill=("time", "max"),
    fillCount=("tid", "size"),
    activeDays=("fillDate", "nunique"),
    activeMonths=("fillMonth", "nunique"),
    totalNotional=("notional", "sum"),
    medianNotional=("notional", "median"),
    totalFee=("fee", "sum"),
    accountValue=("accountValue", "first"),
    allTimeVlm=("allTimeVlm", "first"),
    allTimeRoi=("allTimeRoi", "first"),
    allTimePnl=("allTimePnl", "first"),
).reset_index()

addressFeatures["activeSpanDays"] = (
    addressFeatures["lastFill"] - addressFeatures["firstFill"]
).dt.days.clip(lower=1)
addressFeatures["fillsPerActiveDay"] = addressFeatures["fillCount"] / addressFeatures["activeDays"].replace(0, np.nan)
addressFeatures["notionalPerActiveDay"] = addressFeatures["totalNotional"] / addressFeatures["activeDays"].replace(0, np.nan)
addressFeatures["feeAsPctOfNotional"] = addressFeatures["totalFee"] / addressFeatures["totalNotional"].replace(0, np.nan)

closedAgg = closedDf.groupby(["statusTier", "address"]).agg(
    closedTradeCount=("closedPnl", "size"),
    realizedPnl=("closedPnl", "sum"),
    winRate=("isWin", "mean"),
    avgWin=("closedPnl", lambda s: s[s > 0].mean()),
    avgLoss=("closedPnl", lambda s: s[s < 0].mean()),
).reset_index()

addressFeatures = addressFeatures.merge(closedAgg, on=["statusTier", "address"], how="left")
addressFeatures["closedTradeShare"] = addressFeatures["closedTradeCount"] / addressFeatures["fillCount"]

now = fillsDf["time"].max()
addressFeatures["daysSinceLastFill"] = (now - addressFeatures["lastFill"]).dt.days

print("Address-level behavior summary by statusTier:")
summaryCols = [
    "fillCount",
    "activeDays",
    "activeMonths",
    "fillsPerActiveDay",
    "totalNotional",
    "realizedPnl",
    "winRate",
    "daysSinceLastFill",
]
print(addressFeatures.groupby("statusTier")[summaryCols].median().round(3))

# ---------------------------------------------------------------------------
# 2. Pareto concentration: how much activity/PnL comes from the most active
#    addresses inside each tier?
# ---------------------------------------------------------------------------
def paretoShare(group, valueCol, topFrac):
    values = group[valueCol].fillna(0).sort_values(ascending=False)
    n = max(1, int(np.ceil(len(values) * topFrac)))
    denom = values.sum()
    if denom == 0:
        return np.nan
    return values.iloc[:n].sum() / denom

paretoRows = []
for tier, group in addressFeatures.groupby("statusTier"):
    paretoRows.append({
        "statusTier": tier,
        "top10pctNotionalShare": paretoShare(group, "totalNotional", 0.10),
        "top20pctNotionalShare": paretoShare(group, "totalNotional", 0.20),
        "top10pctFillShare": paretoShare(group, "fillCount", 0.10),
        "top20pctFillShare": paretoShare(group, "fillCount", 0.20),
    })
paretoDf = pd.DataFrame(paretoRows)
print("\nPareto concentration by statusTier:")
print(paretoDf.set_index("statusTier").round(3))

# ---------------------------------------------------------------------------
# 3. Activity funnel: share of sampled addresses crossing increasing activity
#    thresholds in observed fills.
# ---------------------------------------------------------------------------
thresholds = [1, 10, 100, 1000, 5000, 10000]
funnelRows = []
for tier, group in addressFeatures.groupby("statusTier"):
    row = {"statusTier": tier, "addresses": len(group)}
    for threshold in thresholds:
        row[f"fillsGte{threshold}"] = (group["fillCount"] >= threshold).mean()
    funnelRows.append(row)
funnelDf = pd.DataFrame(funnelRows)
print("\nFill-count funnel by statusTier (share of sampled addresses):")
print(funnelDf.set_index("statusTier").round(3))

# ---------------------------------------------------------------------------
# 4. Cohort retention proxy: first-fill month cohort vs active month offset.
#    This is fill-observed retention, not exchange-level account retention.
# ---------------------------------------------------------------------------
addressMonth = fillsDf[["statusTier", "address", "fillMonth"]].drop_duplicates()
firstMonth = addressMonth.groupby(["statusTier", "address"])["fillMonth"].min().rename("cohortMonth").reset_index()
addressMonth = addressMonth.merge(firstMonth, on=["statusTier", "address"], how="left")
addressMonth["fillMonthPeriod"] = pd.PeriodIndex(addressMonth["fillMonth"], freq="M")
addressMonth["cohortMonthPeriod"] = pd.PeriodIndex(addressMonth["cohortMonth"], freq="M")
addressMonth["monthOffset"] = (
    (addressMonth["fillMonthPeriod"].dt.year - addressMonth["cohortMonthPeriod"].dt.year) * 12
    + (addressMonth["fillMonthPeriod"].dt.month - addressMonth["cohortMonthPeriod"].dt.month)
)

cohortCounts = addressMonth.groupby(["statusTier", "cohortMonth", "monthOffset"])["address"].nunique().reset_index(name="activeAddresses")
cohortSizes = firstMonth.groupby(["statusTier", "cohortMonth"])["address"].nunique().reset_index(name="cohortSize")
cohortRetention = cohortCounts.merge(cohortSizes, on=["statusTier", "cohortMonth"], how="left")
cohortRetention["retention"] = cohortRetention["activeAddresses"] / cohortRetention["cohortSize"]

retentionSummary = (
    cohortRetention[cohortRetention["monthOffset"].isin([1, 3, 6])]
    .groupby(["statusTier", "monthOffset"])["retention"]
    .mean()
    .unstack()
)
retentionSummary.columns = [f"m{int(c)}Retention" for c in retentionSummary.columns]
print("\nAverage cohort retention by statusTier:")
print(retentionSummary.round(3))

# ---------------------------------------------------------------------------
# 5. Save derived Layer 1 tables for later Layer 2 feature engineering.
# ---------------------------------------------------------------------------
addressFeatures.to_parquet(f"{OUTPUT_DIR}/addressBehaviorFeatures.parquet", index=False)
paretoDf.to_parquet(f"{OUTPUT_DIR}/paretoByTier.parquet", index=False)
funnelDf.to_parquet(f"{OUTPUT_DIR}/fillCountFunnelByTier.parquet", index=False)
cohortRetention.to_parquet(f"{OUTPUT_DIR}/cohortRetentionByTier.parquet", index=False)

print("\nDone. Layer 1 behavior tables saved to output/layer1Behavior/")
