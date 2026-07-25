import os

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

INPUT_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
OUTPUT_DIR = "output/layer1Behavior"
FILL_CAP = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

fillsDf = pd.read_parquet(INPUT_PATH, columns=["address", "statusTier", "tid"])

fillCounts = (
    fillsDf.groupby(["statusTier", "address"])["tid"]
    .size()
    .rename("fillCount")
    .reset_index()
)
fillCounts["isCapped"] = fillCounts["fillCount"] >= FILL_CAP

print(f"Total addresses with fills: {len(fillCounts)}")
print(f"Capped addresses (fillCount >= {FILL_CAP}): {fillCounts['isCapped'].sum()} "
      f"({fillCounts['isCapped'].mean():.1%})")

byTier = fillCounts.groupby("statusTier")["isCapped"].agg(["sum", "count", "mean"])
byTier.columns = ["cappedCount", "addressCount", "cappedShare"]
print("\nCapped-address share by statusTier:")
print(byTier.round(3))

fillCounts.to_parquet(f"{OUTPUT_DIR}/cappedAddressFlags.parquet", index=False)
print(f"\nSaved flags to {OUTPUT_DIR}/cappedAddressFlags.parquet")