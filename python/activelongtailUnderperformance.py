
import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

FILLS_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
FEATURES_PATH = "output/layer1Behavior/addressBehaviorFeatures.parquet"
OUTPUT_DIR = "output/layer1Behavior"

os.makedirs(OUTPUT_DIR, exist_ok=True)

fillsDf = pd.read_parquet(FILLS_PATH)
featuresDf = pd.read_parquet(FEATURES_PATH)

isEntryFill = fillsDf["dir"].isin(["Open Long", "Open Short"])
isClosedFill = fillsDf["closedPnl"] != 0

# ---------------------------------------------------------------------------
# 1. Position sizing relative to account equity ("leverage" proxy).
# ---------------------------------------------------------------------------
sizingByAddress = fillsDf.groupby(["statusTier", "address"]).apply(
    lambda g: pd.Series({
        "medianNotionalOverAccountValue": (g["notional"] / g["accountValue"].replace(0, np.nan)).median(),
        "meanNotionalOverAccountValue": (g["notional"] / g["accountValue"].replace(0, np.nan)).mean(),
    }),
    include_groups=False,
).reset_index()

# ---------------------------------------------------------------------------
# 2. Entry timing / execution quality: taker (crossed) share, overall and on
#    entry fills specifically.
# ---------------------------------------------------------------------------
executionByAddress = fillsDf.groupby(["statusTier", "address"]).agg(
    takerShareAllFills=("crossed", "mean"),
).reset_index()

entryExecution = (
    fillsDf[isEntryFill]
    .groupby(["statusTier", "address"])
    .agg(takerShareEntries=("crossed", "mean"), entryFillCount=("crossed", "size"))
    .reset_index()
)
executionByAddress = executionByAddress.merge(entryExecution, on=["statusTier", "address"], how="left")

# ---------------------------------------------------------------------------
# 3. Realized edge split by execution type, on closed (PnL-realizing) fills.
# ---------------------------------------------------------------------------
closedDf = fillsDf[isClosedFill].copy()
edgeByExecution = (
    closedDf.groupby(["statusTier", "crossed"])["closedPnl"]
    .agg(medianPnl="median", meanPnl="mean", n="size", winRate=lambda s: (s > 0).mean())
    .reset_index()
)

# ---------------------------------------------------------------------------
# 4. Combine into one address-level table and summarize by tier.
# ---------------------------------------------------------------------------
combined = featuresDf.merge(sizingByAddress, on=["statusTier", "address"], how="left")
combined = combined.merge(executionByAddress, on=["statusTier", "address"], how="left")

summaryCols = [
    "medianNotionalOverAccountValue",
    "takerShareAllFills",
    "takerShareEntries",
    "winRate",
    "avgWin",
    "avgLoss",
]
tierSummary = combined.groupby("statusTier")[summaryCols].median()

print("Position sizing (median notional / accountValue) and execution style, by statusTier (medians):")
print(tierSummary.round(4))

print("\nRealized PnL per closed fill, by statusTier and execution type "
      "(crossed=True is taker/aggressive, crossed=False is maker/passive):")
print(edgeByExecution.set_index(["statusTier", "crossed"]).round(3))

# ---------------------------------------------------------------------------
# 5. Direct comparison: activeLongTail vs activeCore on the two candidate
#    mechanisms, since activeCore is the tier activeLongTail is falling short
#    of despite similar recency/scale-adjusted activity.
# ---------------------------------------------------------------------------
compareTiers = tierSummary.loc[["activeCore", "activeLongTail"]]
print("\nactiveCore vs activeLongTail, direct comparison:")
print(compareTiers.round(4).T)

combined.to_parquet(f"{OUTPUT_DIR}/leverageAndExecutionByAddress.parquet", index=False)
tierSummary.reset_index().to_parquet(f"{OUTPUT_DIR}/leverageAndExecutionByTier.parquet", index=False)

print(f"\nSaved to {OUTPUT_DIR}/")