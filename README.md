### Address from https://stats-data.hyperliquid.xyz/Mainnet/leaderboard.
Excludes accounts with less than 100k USDC account value and less than 10M USDC trading volume.

### Conclusions from the Distribution Analysis
[jump to Leaderboard Cell](./python/leaderboardFeaturing.ipynb#cell-66171ff7)


Based on the printed statistics, percentile breakdowns, and normality tests produced by analyzeAllMetrics(), here are the key findings for the 40,471-account Hyperliquid leaderboard dataset.

1. Extreme Right-Skew Across All Metrics

Every single metric — accountValue, all four vlm windows, all four pnl windows, and all four roi windows — shows a massive gap between the median and the mean, driven by a small number of outlier accounts:

Metric	        Median (P50)	    Mean	        Max
accountValue	995.41	            992,527.59	    13,267,659,935.89
all_time_vlm	20,709,868.47	    243,797,441.39	721,659,936,506.50
all_time_pnl	-250.69	            319,901.01	    445,047,889.91
all_time_roi	-0.0043	            44.96	        168,656.51

The mean of all_time_roi (44.96, i.e. ~4,496% average return) versus a median close to zero illustrates that a handful of extreme winners are dragging the average far above what a typical trader experiences.

2. Most Accounts Are Inactive or Dormant

For the shorter time windows, a very large share of accounts show exactly zero activity:

day_vlm: P1–P75 are all 0.00 — meaning at least 75% of accounts traded zero volume in the last day.
week_vlm: P1–P50 are 0.00 — at least half of accounts had no trades in the past week.
month_vlm: P1–P25 are 0.00.

This confirms the dataset is dominated by inactive or "ghost" wallets, and only a minority of addresses are actively trading in any given short window. Any modeling that treats the full population as "active traders" would be misleading.

3. Log-Transformation Helps, But Doesn't Achieve Normality

Log1p / signed-log transforms substantially compress the scale (e.g., accountValue log-mean 7.21 vs. raw mean 992,527), but none of the 13 metrics pass normality tests:

Every Shapiro-Wilk p-value is effectively 0 (largest is all_time_pnl at p ≈ 7.77e-62).
Every KS test p-value is also ≈ 0.

So while log-normal is a reasonable approximation for magnitude-only variables like accountValue and the vlm series (kurtosis close to 0–1, i.e. near mesokurtic after transform), it is not statistically valid as an exact distributional assumption. Treat log-normal fits as a working approximation, not ground truth.

4. Short-Window ROI Metrics Have Pathological Tails

The roi metrics get progressively worse (more leptokurtic) as the time window shrinks:

Metric	Log Skew	Log Kurtosis
month_roi	2.37	48.85
week_roi	7.10	206.93
day_roi	0.18	124.09
all_time_roi	2.83	12.06

A kurtosis of 206.93 for week_roi (vs. 3 for a normal distribution) indicates an extremely fat-tailed distribution with rare but enormous outliers (max week_roi = 355.63, i.e. a ~35,563% weekly return for at least one account). These are likely small-account artifacts (tiny denominators inflating ROI) rather than representative trading skill, and should be capped, winsorized, or filtered by a minimum account-value / volume threshold before use in any downstream model.

5. PnL Is Symmetric in Direction but Asymmetric in Magnitude
Log-skew for pnl metrics is close to 0 (e.g., all_time_pnl: 0.091, week_pnl: -0.0009), meaning wins and losses are roughly balanced in frequency after the signed-log transform.
However, raw-scale kurtosis and the P1/P99 percentile gaps (e.g. all_time_pnl P1 = -2,024,300 vs P99 = +7,455,044) show wins can be several times larger in magnitude than losses at the extremes — consistent with a market where top traders capture disproportionate gains.

### EDA

#### raw output
Total fills loaded (post-dedup): 2,084,221
Unique addresses in fills: 302 / 320 sampled

Addresses with ZERO fills returned: 18
statusTier
silentHolder    18
dtype: int64

Fill count distribution by statusTier:
                count      mean          std     min      25%     50%      75%      max
statusTier                                                                             
activeCore      100.0  7031.440  3956.379182    36.0  3287.00  9962.0   9999.0  10000.0
activeLongTail   80.0  7272.350  3428.535315  1234.0  3425.50  9982.0   9999.0  10000.0
churnedTrader   100.0  6634.280  3711.358079    26.0  2854.75  8589.5  10000.0  10000.0
silentHolder     40.0  3396.525  3999.433146     0.0     0.00   756.5   7458.0  10000.0
E:\projects\HlTraderAction\python\fillsEDA.py:101: Pandas4Warning: Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now('UTC') instead.
  now = pd.Timestamp.utcnow().tz_localize(None)

Days since last fill, by statusTier (this is the core churn signal):
                count        mean         std   min     25%    50%     75%    max
statusTier                                                                       
activeCore      100.0  158.260000  181.490775   2.0   12.50   89.0  277.75  771.0
activeLongTail   80.0  149.825000  192.155513   2.0    5.75   38.5  268.25  802.0
churnedTrader   100.0  321.570000  196.214439  41.0  169.25  298.0  419.50  866.0
silentHolder     22.0  173.318182  155.348268   4.0   43.25  131.5  289.50  490.0

Notional per fill, by statusTier:
                   count          mean           std       min         25%          50%          75%           max
statusTier                                                                                                        
activeCore      703144.0   7793.298787  43310.329785  0.000000   95.573415   663.423820  3142.335127  1.070860e+07
activeLongTail  581788.0   9171.037807  46500.936963  0.000000  105.060171   769.309045  3972.173935  7.958993e+06
churnedTrader   663428.0  13012.433028  70512.493436  0.003047  370.378771  1412.909011  6796.000000  2.407871e+07
silentHolder    135861.0   6301.279525  25985.155198  0.000000  200.054800   857.080500  2999.978856  2.110302e+06

Buy/sell side mix by statusTier (share of fills):
side                A      B
statusTier                  
activeCore      0.489  0.511
activeLongTail  0.491  0.509
churnedTrader   0.481  0.519
silentHolder    0.484  0.516

Trade direction mix by statusTier (open/close, long/short):
dir             Auto-Deleveraging    Buy  Close Long  Close Short  ...  Settlement  Short > Long  Split Outcome  Spot Dust ConversionstatusTier                                                         ...                                                               
activeCore                    0.0  0.083       0.297        0.114  ...         0.0         0.001            0.0                   0.0activeLongTail                0.0  0.032       0.272        0.174  ...         0.0         0.003            0.0                   0.0churnedTrader                 0.0  0.044       0.268        0.156  ...         0.0         0.001            0.0                   0.0silentHolder                  0.0  0.036       0.327        0.095  ...         0.0         0.000            0.0                   0.0
[4 rows x 16 columns]

Win rate & average win/loss, by statusTier:
                winRate   avgWin  avgLoss
statusTier                               
activeCore        0.543  238.664 -344.615
activeLongTail    0.510  175.851 -104.129
churnedTrader     0.559  431.690 -270.032
silentHolder      0.469  145.169 -115.022

Fee burden by statusTier:
                    totalFee  totalNotional  feeAsPctOfNotional
statusTier                                                     
activeCore      1.849065e+06   5.479811e+09            0.000337
activeLongTail  1.576123e+06   5.335600e+09            0.000295
churnedTrader   2.550258e+06   8.632812e+09            0.000295
silentHolder    2.985407e+05   8.560981e+08            0.000349

Average daily fill count on active days, by statusTier:
statusTier
activeCore         76.06
activeLongTail     92.45
churnedTrader     295.51
silentHolder       62.87
Name: dailyFills, dtype: float64

#### Explanation

1. Sample coverage
320 addresses sampled, fills retrieved for 302 (94.4%). 18 addresses returned zero fills, all under `silentHolder` — consistent with the label, but means several `silentHolder` stats below are computed on small sub-samples (n=22 for recency, n=40 for others). Treat those numbers as directional, not robust.

2. Fill volume is not a good discriminator
| Tier | Mean fills | Median | Max |
|---|---:|---:|---:|
| activeCore | 7,031.4 | 9,962 | 10,000 |
| activeLongTail | 7,272.4 | 9,982 | 10,000 |
| churnedTrader | 6,634.3 | 8,589.5 | 10,000 |
| silentHolder | 3,396.5 | 756.5 | 10,000 |

`activeCore`, `activeLongTail`, and `churnedTrader` look nearly identical here — medians all near the 10,000-fill cap. Only `silentHolder` is clearly distinguished, and even there the mean is inflated by a right tail — the median (756.5) is the more honest summary. **Takeaway: tier separation is not coming from historical activity volume.**

3. Recency cleanly validates the churn narrative
| Tier | Mean days since last fill | Median |
|---|---:|---:|
| activeCore | 158.3 | 89.0 |
| activeLongTail | 149.8 | 38.5 |
| churnedTrader | 321.6 | **298.0** |
| silentHolder | 173.3 | 131.5 |

`churnedTrader` sits well apart from the other three (median 38–132 days) with a median of 298 days inactive — roughly 3–8x longer. This is the cleanest split in the dataset and cross-validates the original tiering logic. Worth a side note: `activeCore` and `activeLongTail` have similar means but very different medians (89 vs 38.5), meaning `activeCore` has its own long tail of semi-stale accounts pulling the mean up — it's not as uniformly "active" as the label suggests.

4. The real story: high win rate + high notional ≠ healthy churned traders
| Tier | Win rate | Avg win | Avg loss | Notional/fill (mean) | Daily fills (active days) |
|---|---:|---:|---:|---:|---:|
| activeCore | 54.3% | 238.7 | -344.6 | 7,793 | 76.1 |
| activeLongTail | 51.0% | 175.9 | -104.1 | 9,171 | 92.5 |
| churnedTrader | **55.9%** | **431.7** | -270.0 | **13,012** | **295.5** |
| silentHolder | 46.9% | 145.2 | -115.0 | 6,301 | 62.9 |

`churnedTrader` leads on every "performance-looking" metric: highest win rate, largest average win, largest per-fill notional, and 3–4x the daily fill count of any other tier. Read naively, this looks like your best-performing segment.

That reading is wrong. This combination — high frequency, high win rate, growing position size — is the classic **"picking up pennies in front of a steamroller"** pattern. Frequent small wins look great on a mean/win-rate basis, but they're structurally vulnerable to a single large tail event (most plausibly a liquidation) that wipes out the accumulated gains in one shot, after which the account goes dark. Mean and win-rate statistics are exactly the kind of summary that hides this — they're computed on the surviving history, not on the terminal event that ended it.

The earlier EDA pass didn't surface ROI, only win/loss averages. If churned-trader median ROI turns out negative (the figure ~-47.7% was flagged separately), that's the piece that would lock in the steamroller interpretation over an alternative "traded well and walked away" story. **This is the single most useful number to pull next.**

5. Fees don't explain `activeLongTail`'s underperformance
Fee-as-%-of-notional is flat across all four tiers (0.0295%–0.0349%), with no tier standing out. So whatever is driving `activeLongTail`'s comparatively weak average win (175.9, lowest among the three "active-ish" tiers) and moderate win rate (51.0%) isn't fee drag. More likely candidates: gradually increasing leverage, worse entry timing, or position sizing that doesn't scale with edge — worth checking trends over time within this tier specifically.


#### Churn investigation


1. Liquidation is not the churn driver
Only 76 fills across the entire sample are liquidation/ADL-tagged (out of ~2.08M total fills), and they're not concentrated in `churnedTrader`:

| Tier | Liquidation fills | Sample size |
|---|---:|---:|
| activeCore | 46 | 100 |
| activeLongTail | 15 | 80 |
| churnedTrader | 13 | 100 |
| silentHolder | 2 | ~40 |

`activeCore` actually has the most liquidation events, both in absolute count and relative to sample size — `churnedTrader` has fewer. **This directly rejects the "steamroller" hypothesis from the previous round.** Churned traders are not, on the whole, exiting because they got liquidated.

2. Tail-loss concentration is similar across tiers, not uniquely severe for churned
"Share of total loss from the single worst trade" (higher = more tail-driven):

| Tier | Median | Mean |
|---|---:|---:|
| activeCore | 0.066 | 0.133 |
| activeLongTail | 0.047 | 0.078 |
| churnedTrader | 0.048 | 0.126 |
| silentHolder | 0.052 | 0.114 |

Medians are all in a tight 0.047–0.066 range. `churnedTrader`'s mean is pulled up by outliers (max = 1.00, meaning at least one account's entire loss came from one trade), but the median shows this is not the typical pattern for the tier — it's a tail case, not the tier's defining behavior.

3. Position escalation is modest for churned traders — `activeCore` is the outlier
"Last 20% notional vs first 20% notional" (>1 = growing position size over the account's history):

| Tier | Median | Mean | Max |
|---|---:|---:|---:|
| activeCore | 1.040 | 4.186 | **174.6** |
| activeLongTail | 1.021 | 1.567 | 23.3 |
| churnedTrader | 1.090 | 1.701 | 27.9 |
| silentHolder | 1.519 | 2.358 | 12.8 |

Medians are all close to 1 (roughly stable sizing), but `activeCore`'s mean and max are far higher than any other tier — it has the most extreme escalation cases, not `churnedTrader`. This further undercuts the earlier "churned traders grew reckless before blowing up" narrative.

4. Final-week intensity: churned traders slow down, they don't speed up
"Final week fill rate vs. account's overall average" (>1 = sped up before going quiet):

| Tier | Median | Mean |
|---|---:|---:|
| activeCore | 0.971 | 2.476 |
| activeLongTail | 0.559 | 2.031 |
| churnedTrader | **0.714** | 1.340 |
| silentHolder | 1.032 | 2.613 |

This is the clearest finding of this round. `churnedTrader`'s median is 0.714 — **below 1**, meaning the typical churned account was trading *less* than its own historical average in its final week, not ramping up into a blowup. `churnedTrader` also has the lowest mean of any tier. Combined with the low liquidation rate, this points toward a **gradual wind-down / fading disengagement** pattern rather than a sudden forced exit.

## Revised interpretation
The "picking up pennies in front of a steamroller" story from the last round doesn't hold up:
- Liquidations are rare overall and not concentrated in `churnedTrader`.
- Tail-loss concentration is similar across tiers.
- Position escalation is unremarkable for `churnedTrader` (it's `activeCore` that shows extreme escalation, yet that tier remains active).
- Churned traders slow down before going dark, they don't speed up.

The more consistent story now is **gradual disengagement**: churned traders taper off trading intensity over time and eventually stop, rather than getting wiped out in a single event. What's still unexplained is *why* — worth checking PnL trend over the account's lifetime (declining edge? losing streak without a single catastrophic trade?) and whether tapering correlates with rising loss rate rather than one big loss.

