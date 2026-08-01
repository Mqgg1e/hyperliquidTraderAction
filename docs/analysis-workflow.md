# Analysis workflow

Execution order and file traceability for the full pipeline. Every script reads from
`output/` and/or ClickHouse and writes new parquet files to `output/`; run in this order
since later steps depend on earlier ones.

## 0. Data ingestion (one-time / re-run to refresh)

| Script | Reads | Writes |
|---|---|---|
| `python/fetchSave.py` | Hyperliquid `userFillsByTime` API | ClickHouse `hyperliquid.fills`, `output/hyperliquidLeaderboard.parquet`, `output/addresses_sample.parquet` / `.json` |
| `python/leaderboardSampling.ipynb`, `python/leaderboardFeaturing.ipynb`, `python/leaderboardAddress.ipynb` | `output/hyperliquidLeaderboard.parquet` | `statusTier` tiering logic, stratified sample (notebooks — the "Conclusions from the Distribution Analysis" and tier definitions in the findings doc come from these) |

`python/ufbtTest.py` is a standalone scratch script for testing the `userFillsByTime` endpoint directly — not part of the output pipeline, no parquet in/out.

## 1. EDA and address-level feature engineering

| Order | Script | Reads | Writes |
|---|---|---|---|
| 1 | `python/fillsEDA.py` | `output/addresses_sample.parquet`, ClickHouse `fills` | `output/edaOutputs/fills_merged_with_tier.parquet`, `address_time_span.parquet` |
| 2 | `python/layer1BehaviorAnalysis.py` | `fills_merged_with_tier.parquet` | `output/layer1Behavior/addressBehaviorFeatures.parquet`, `paretoByTier.parquet`, `fillCountFunnelByTier.parquet`, `cohortRetentionByTier.parquet` |

## 2. Churn mechanism investigation

| Order | Script | Reads | Writes | Status |
|---|---|---|---|---|
| 3 | `python/churnInv.py` | `fills_merged_with_tier.parquet` | `output/churnMechanism/lossConcentrationByAddress.parquet`, `escalationByAddress.parquet`, `finalWeekIntensityByAddress.parquet` | rules out liquidation, single catastrophic trade, and escalation-into-blowup |
| 4 | `python/pnlInv.py` | `fills_merged_with_tier.parquet` | `output/churnMechanism/pnlQuintileByAddress.parquet`, `winRateSlopeByAddress.parquet` | superseded approach (trade-count quintiles, calendar-alignment confound) — kept for the investigation trail, not cited in conclusions |
| 5 | `python/churnedEdgeDecline.py` | `fills_merged_with_tier.parquet` | `output/churnMechanism/netPnlByAddressMonthBucket.parquet`, `shareNetProfitableByTierBucket.parquet` | the approach that actually resolved the "why does churn happen" question — address-relative month buckets + share-profitable metric |

## 3. Data-quality correction (10k fill-cap)

| Order | Script | Reads | Writes |
|---|---|---|---|
| 6 | `python/identifyCappedAddress.py` | `fills_merged_with_tier.parquet` | `output/layer1Behavior/cappedAddressFlags.parquet` |
| 7 | `python/cappedAddressCorrection.py` | `addressBehaviorFeatures.parquet`, `cappedAddressFlags.parquet`, `fills_merged_with_tier.parquet` | `output/layer1Behavior/activeSpanCorrectedByTier.parquet`, `cohortRetentionCorrectedByTier.parquet`, `cohortRetentionCorrectedSummaryByTier.parquet` |

Can technically run any time after step 2, but conceptually belongs before trusting any `firstFill`-derived number in steps 1-2's raw output.

## 4. activeLongTail underperformance

| Order | Script | Reads | Writes |
|---|---|---|---|
| 8 | `python/activelongtailUnderperformance.py` | `fills_merged_with_tier.parquet`, `addressBehaviorFeatures.parquet` | `output/layer1Behavior/leverageAndExecutionByAddress.parquet`, `leverageAndExecutionByTier.parquet` |

## 5. Layer 2 — clustering

| Order | Script | Reads | Writes |
|---|---|---|---|
| 9 | `python/layer2clustering.py` | `leverageAndExecutionByAddress.parquet`, `cappedAddressFlags.parquet` | `output/layer2Clustering/clusterAssignments.parquet`, `clusterVsTierCrosstab.parquet`, `churnedTraderSubclusters.parquet` |
| 10 | `python/churnedSubcluster.py` | `fills_merged_with_tier.parquet`, `churnedTraderSubclusters.parquet`, `leverageAndExecutionByTier.parquet` | `output/layer2Clustering/churnedSubclusterExecutionMechanism.parquet` |

## 6. Presentation layer

| Script | Depends on | Run with |
|---|---|---|
| `dashboard/app.py` | every parquet output from steps 1-5 above (reads them live, nothing hardcoded) | `uv run streamlit run dashboard/app.py` |

## Notes

- All `output/*.parquet` files are generated artifacts, intentionally not tracked in git — re-run the pipeline above to recreate them locally.
- Steps within the same numbered section (e.g. 3-5 in Section 2) don't depend on each other and can run in any order relative to one another, but all of Section 1 must run first, and Section 3/4/5 all depend on Section 1's `addressBehaviorFeatures.parquet` and/or Section 2/3's outputs as noted in the "Reads" column.
- `python/pnlInv.py` (step 4) is kept as-is for traceability even though its approach was superseded by step 5 — see `docs/findings.md` under "pnl investigation" for why.