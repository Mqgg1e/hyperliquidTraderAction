import os

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

FEATURES_PATH = "output/layer1Behavior/leverageAndExecutionByAddress.parquet"
FLAGS_PATH = "output/layer1Behavior/cappedAddressFlags.parquet"
OUTPUT_DIR = "output/layer2Clustering"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(FEATURES_PATH)
flagsDf = pd.read_parquet(FLAGS_PATH)[["statusTier", "address", "isCapped"]]
df = df.merge(flagsDf, on=["statusTier", "address"], how="left")

# ---------------------------------------------------------------------------
# 1. Impute the small number of NaNs (addresses with no closed trades / no
#    entry fills) with the address's own tier median, then build log-scaled
#    features for the heavy-tailed columns.
# ---------------------------------------------------------------------------
imputeCols = ["winRate", "avgWin", "avgLoss", "closedTradeShare", "takerShareEntries"]
for col in imputeCols:
    df[col] = df.groupby("statusTier")[col].transform(lambda s: s.fillna(s.median()))

featureMatrix = pd.DataFrame(index=df.index)
featureMatrix["fillCountLog"] = np.log1p(df["fillCount"])
featureMatrix["totalNotionalLog"] = np.log1p(df["totalNotional"])
featureMatrix["fillsPerActiveDayLog"] = np.log1p(df["fillsPerActiveDay"])
featureMatrix["notionalPerActiveDayLog"] = np.log1p(df["notionalPerActiveDay"])
featureMatrix["feeAsPctOfNotional"] = df["feeAsPctOfNotional"]
featureMatrix["winRate"] = df["winRate"]
featureMatrix["avgWinLog"] = np.log1p(df["avgWin"].clip(lower=0))
featureMatrix["avgLossAbsLog"] = np.log1p(df["avgLoss"].abs())
featureMatrix["closedTradeShare"] = df["closedTradeShare"]
featureMatrix["daysSinceLastFillLog"] = np.log1p(df["daysSinceLastFill"])
featureMatrix["takerShareAllFills"] = df["takerShareAllFills"]
featureMatrix["takerShareEntries"] = df["takerShareEntries"]

scaler = StandardScaler()
scaledMatrix = scaler.fit_transform(featureMatrix.values)

# ---------------------------------------------------------------------------
# 2. KMeans, k=4 (to compare directly against the four statusTier labels),
#    plus a silhouette sweep over k=2..7 to check whether 4 is actually the
#    best-supported number of clusters in the feature space.
# ---------------------------------------------------------------------------
print("Silhouette score by k (KMeans):")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(scaledMatrix)
    print(f"  k={k}: silhouette={silhouette_score(scaledMatrix, labels):.3f}")

kmeans4 = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
df["kmeansCluster"] = kmeans4.fit_predict(scaledMatrix)

ariKmeans = adjusted_rand_score(df["statusTier"], df["kmeansCluster"])
print(f"\nKMeans (k=4) vs statusTier, adjusted Rand index: {ariKmeans:.3f}")

crosstabKmeans = pd.crosstab(df["statusTier"], df["kmeansCluster"])
print("\nKMeans cluster vs statusTier crosstab:")
print(crosstabKmeans)

# ---------------------------------------------------------------------------
# 3. HDBSCAN, to see whether a density-based method (no forced k) recovers a
#    similar structure or something different, and to see how much it labels
#    as noise.
# ---------------------------------------------------------------------------
hdbscanModel = HDBSCAN(min_cluster_size=15, min_samples=5)
df["hdbscanCluster"] = hdbscanModel.fit_predict(scaledMatrix)

noiseShare = (df["hdbscanCluster"] == -1).mean()
nClustersHdbscan = df.loc[df["hdbscanCluster"] != -1, "hdbscanCluster"].nunique()
print(f"\nHDBSCAN: {nClustersHdbscan} clusters found, {noiseShare:.1%} of addresses labeled noise")

ariHdbscan = adjusted_rand_score(df["statusTier"], df["hdbscanCluster"])
print(f"HDBSCAN vs statusTier, adjusted Rand index: {ariHdbscan:.3f}")

crosstabHdbscan = pd.crosstab(df["statusTier"], df["hdbscanCluster"])
print("\nHDBSCAN cluster vs statusTier crosstab:")
print(crosstabHdbscan)

# ---------------------------------------------------------------------------
# 4. Cluster centroid profile (KMeans) in the original feature units, to
#    interpret what each cluster represents.
# ---------------------------------------------------------------------------
profileCols = ["fillCount", "fillsPerActiveDay", "daysSinceLastFill", "winRate",
               "avgWin", "avgLoss", "takerShareAllFills", "notionalPerActiveDay"]
clusterProfile = df.groupby("kmeansCluster")[profileCols].median()
clusterProfile["addressCount"] = df.groupby("kmeansCluster").size()
clusterProfile["dominantTier"] = df.groupby("kmeansCluster")["statusTier"].agg(lambda s: s.value_counts().idxmax())
print("\nKMeans cluster profile (medians):")
print(clusterProfile.round(2))

# ---------------------------------------------------------------------------
# 5. Within churnedTrader only: is there a fast-decay vs. slow-decay axis?
#    Re-cluster just this tier's addresses on intensity/recency features.
# ---------------------------------------------------------------------------
churnedDf = df[df["statusTier"] == "churnedTrader"].copy()
churnedFeatureCols = ["fillsPerActiveDayLog", "daysSinceLastFillLog", "winRate",
                       "avgWinLog", "avgLossAbsLog", "takerShareAllFills"]
churnedFeatureIdx = [featureMatrix.columns.get_loc(c) for c in churnedFeatureCols]
churnedScaled = StandardScaler().fit_transform(featureMatrix.loc[churnedDf.index, churnedFeatureCols].values)

print("\nSilhouette score by k, churnedTrader-only re-clustering:")
for k in range(2, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(churnedScaled)
    print(f"  k={k}: silhouette={silhouette_score(churnedScaled, labels):.3f}")

churnedKmeans = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
churnedDf["subCluster"] = churnedKmeans.fit_predict(churnedScaled)

subClusterProfile = churnedDf.groupby("subCluster")[
    ["fillsPerActiveDay", "daysSinceLastFill", "winRate", "avgWin", "avgLoss", "takerShareAllFills"]
].median()
subClusterProfile["addressCount"] = churnedDf.groupby("subCluster").size()
print("\nchurnedTrader sub-cluster profile (medians):")
print(subClusterProfile.round(2))

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
df[["statusTier", "address", "isCapped", "kmeansCluster", "hdbscanCluster"]].to_parquet(
    f"{OUTPUT_DIR}/clusterAssignments.parquet", index=False
)
crosstabKmeans.to_parquet(f"{OUTPUT_DIR}/clusterVsTierCrosstab.parquet")
churnedDf[["address", "subCluster"]].to_parquet(f"{OUTPUT_DIR}/churnedTraderSubclusters.parquet", index=False)

print(f"\nSaved to {OUTPUT_DIR}/")