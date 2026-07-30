import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

FILLS_PATH = "output/edaOutputs/fills_merged_with_tier.parquet"
SUBCLUSTER_PATH = "output/layer2Clustering/churnedTraderSubclusters.parquet"
TIER_BASELINE_PATH = "output/layer1Behavior/leverageAndExecutionByTier.parquet"
OUTPUT_DIR = "output/layer2Clustering"

fillsDf = pd.read_parquet(FILLS_PATH)
subClusterDf = pd.read_parquet(SUBCLUSTER_PATH)
tierBaseline = pd.read_parquet(TIER_BASELINE_PATH)

churnedFills = fillsDf[fillsDf["statusTier"] == "churnedTrader"].merge(
    subClusterDf, on="address", how="inner"
)

# ---------------------------------------------------------------------------
# 1. Realized PnL per closed fill, by sub-cluster and execution type - same
#    lens used for the activeLongTail check.
# ---------------------------------------------------------------------------
isClosedFill = churnedFills["closedPnl"] != 0
closedDf = churnedFills[isClosedFill]

edgeBySubclusterExecution = (
    closedDf.groupby(["subCluster", "crossed"])["closedPnl"]
    .agg(medianPnl="median", n="size", winRate=lambda s: (s > 0).mean())
    .reset_index()
)
print("churnedTrader realized PnL per closed fill, by sub-cluster and execution type:")
print(edgeBySubclusterExecution.set_index(["subCluster", "crossed"]).round(3))

# ---------------------------------------------------------------------------
# 2. Taker share and sizing scale by sub-cluster (address-level).
# ---------------------------------------------------------------------------
addressLevel = churnedFills.groupby(["address", "subCluster"]).agg(
    takerShareAllFills=("crossed", "mean"),
    medianNotional=("notional", "median"),
).reset_index()

subClusterSummary = addressLevel.groupby("subCluster")[["takerShareAllFills", "medianNotional"]].median()
subClusterSummary["addressCount"] = addressLevel.groupby("subCluster").size()
print("\nchurnedTrader sub-cluster taker share and sizing scale (medians):")
print(subClusterSummary.round(2))

# ---------------------------------------------------------------------------
# 3. Baseline comparison against activeLongTail / activeCore from Layer 1.
# ---------------------------------------------------------------------------
print("\nBaseline for comparison (from leverageAndExecutionByTier.parquet):")
print(tierBaseline.set_index("statusTier")[["takerShareAllFills", "winRate"]].round(3))

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
edgeBySubclusterExecution.to_parquet(f"{OUTPUT_DIR}/churnedSubclusterExecutionMechanism.parquet", index=False)
print(f"\nSaved to {OUTPUT_DIR}/churnedSubclusterExecutionMechanism.parquet")