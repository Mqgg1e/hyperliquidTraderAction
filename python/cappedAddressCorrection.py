
import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

FEATURES_PATH = "output/layer1Behavior/addressBehaviorFeatures.parquet"
FLAGS_PATH = "output/layer1Behavior/cappedAddressFlags.parquet"
FILLS_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
OUTPUT_DIR = "output/layer1Behavior"

os.makedirs(OUTPUT_DIR, exist_ok=True)

featuresDf = pd.read_parquet(FEATURES_PATH)
flagsDf = pd.read_parquet(FLAGS_PATH)[["statusTier", "address", "isCapped"]]
featuresDf = featuresDf.merge(flagsDf, on=["statusTier", "address"], how="left")

# ---------------------------------------------------------------------------
# 1. Active-span correction: full sample vs non-capped-only, by statusTier.
# ---------------------------------------------------------------------------
spanRows = []
for tier, group in featuresDf.groupby("statusTier"):
    nonCapped = group[~group["isCapped"]]
    spanRows.append({
        "statusTier": tier,
        "addressCount": len(group),
        "cappedCount": int(group["isCapped"].sum()),
        "activeSpanDaysFullSampleMedian": group["activeSpanDays"].median(),
        "activeSpanDaysNonCappedMedian": nonCapped["activeSpanDays"].median(),
        "nonCappedAddressCount": len(nonCapped),
    })
spanDf = pd.DataFrame(spanRows)
print("Active span (days): full sample vs non-capped-only, by statusTier")
print(spanDf.set_index("statusTier").round(1))

# ---------------------------------------------------------------------------
# 2. Cohort retention correction: re-derive cohortMonth using ONLY non-capped
#    addresses, so cohortMonth reflects a genuine firstFill rather than an
#    artifact of the 10k-fill window.
# ---------------------------------------------------------------------------
fillsDf = pd.read_parquet(FILLS_PATH, columns=["address", "statusTier", "time"])
fillsDf["time"] = pd.to_datetime(fillsDf["time"])
fillsDf["fillMonth"] = fillsDf["time"].dt.to_period("M").astype(str)

nonCappedAddresses = flagsDf.loc[~flagsDf["isCapped"], ["statusTier", "address"]]
fillsNonCapped = fillsDf.merge(nonCappedAddresses, on=["statusTier", "address"], how="inner")

addressMonth = fillsNonCapped[["statusTier", "address", "fillMonth"]].drop_duplicates()
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
retentionSummary.columns = [f"m{int(c)}RetentionCorrected" for c in retentionSummary.columns]

nonCappedSizePerTier = nonCappedAddresses.groupby("statusTier").size().rename("nonCappedAddressCount")
retentionSummary = retentionSummary.join(nonCappedSizePerTier)

print("\nCohort retention (non-capped-only, corrected) by statusTier:")
print(retentionSummary.round(3))

print("\nSample size warning: cohort-month buckets with < 5 addresses in a "
      "cohort are unreliable and should be flagged, not reported, in the README.")
smallCohorts = cohortSizes[cohortSizes["cohortSize"] < 5]
print(f"Cohorts with < 5 non-capped addresses: {len(smallCohorts)} of {len(cohortSizes)}")

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
spanDf.to_parquet(f"{OUTPUT_DIR}/activeSpanCorrectedByTier.parquet", index=False)
cohortRetention.to_parquet(f"{OUTPUT_DIR}/cohortRetentionCorrectedByTier.parquet", index=False)
retentionSummary.reset_index().to_parquet(f"{OUTPUT_DIR}/cohortRetentionCorrectedSummaryByTier.parquet", index=False)

print(f"\nSaved corrected tables to {OUTPUT_DIR}/")