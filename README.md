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


### Churn investigation
#### Raw output
| Direction | Count |
|-----------|-------|
| Open Long | 657,396 |
| Close Long | 589,548 |
| Open Short | 327,807 |
| Close Short | 297,901 |
| Buy | 111,523 |
| Sell | 92,417 |
| Short > Long | 3,158 |
| Long > Short | 3,116 |
| Merge Outcome | 872 |
| Split Outcome | 188 |
| Settlement | 139 |
| Auto-Deleveraging | 79 |
| Liquidated Isolated Long | 39 |
| Liquidated Cross Long | 36 |
| Liquidated Isolated Short | 1 |
| Spot Dust Conversion | 1 |

> **Name:** count, **dtype:** int64[pyarrow]

---

Fills Tagged as Liquidation-Related

| statusTier | Count |
|------------|-------|
| activeCore | 46 |
| activeLongTail | 15 |
| churnedTrader | 13 |
| silentHolder | 2 |

> **dtype:** int64

---

Share of Total Loss Coming from the Single Worst Trade, by Tier

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 99.0 | 0.132764 | 0.180953 | 0.010328 | 0.034875 | 0.066010 | 0.114957 | 0.912984 |
| activeLongTail | 80.0 | 0.078463 | 0.078153 | 0.003956 | 0.025820 | 0.047326 | 0.103089 | 0.414775 |
| churnedTrader | 88.0 | 0.125616 | 0.197377 | 0.003066 | 0.030540 | 0.048300 | 0.114213 | 1.000000 |
| silentHolder | 21.0 | 0.114061 | 0.216573 | 0.019299 | 0.037834 | 0.051722 | 0.075443 | 1.000000 |

---

Position Size Escalation Ratio (Last 20% Notional / First 20% Notional), by Tier

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 100.0 | 4.185927 | 17.844580 | 0.061412 | 0.648868 | 1.040201 | 1.832589 | 174.605318 |
| activeLongTail | 80.0 | 1.566948 | 2.637714 | 0.151279 | 0.630691 | 1.020823 | 1.740387 | 23.310915 |
| churnedTrader | 100.0 | 1.701443 | 2.971192 | 0.064355 | 0.666402 | 1.090121 | 1.662843 | 27.877815 |
| silentHolder | 20.0 | 2.357716 | 2.872906 | 0.369117 | 0.745005 | 1.518627 | 2.330486 | 12.808838 |

---

Final Week Trading Intensity vs Overall Average, by Tier (>1 = Sped Up Before Going Quiet)

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 100.0 | 2.475545 | 4.192084 | 0.012809 | 0.424644 | 0.971255 | 2.651782 | 30.614408 |
| activeLongTail | 80.0 | 2.031254 | 6.374289 | 0.025241 | 0.230584 | 0.558923 | 1.221223 | 46.580645 |
| churnedTrader | 100.0 | 1.339681 | 2.188494 | 0.002910 | 0.204545 | 0.714286 | 1.235910 | 13.642000 |
| silentHolder | 22.0 | 2.613073 | 4.872185 | 0.054206 | 0.467877 | 1.032121 | 2.779164 | 23.330499 |

---

#### Explanation
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

The "picking up pennies in front of a steamroller" story from the last round doesn't hold up:
- Liquidations are rare overall and not concentrated in `churnedTrader`.
- Tail-loss concentration is similar across tiers.
- Position escalation is unremarkable for `churnedTrader` (it's `activeCore` that shows extreme escalation, yet that tier remains active).
- Churned traders slow down before going dark, they don't speed up.

The more consistent story now is **gradual disengagement**: churned traders taper off trading intensity over time and eventually stop, rather than getting wiped out in a single event. What's still unexplained is *why* — worth checking PnL trend over the account's lifetime (declining edge? losing streak without a single catastrophic trade?) and whether tapering correlates with rising loss rate rather than one big loss.

### pnl investigation
#### Raw output
| Direction | Count |
|-----------|-------|
| Open Long | 657,396 |
| Close Long | 589,548 |
| Open Short | 327,807 |
| Close Short | 297,901 |
| Buy | 111,523 |
| Sell | 92,417 |
| Short > Long | 3,158 |
| Long > Short | 3,116 |
| Merge Outcome | 872 |
| Split Outcome | 188 |
| Settlement | 139 |
| Auto-Deleveraging | 79 |
| Liquidated Isolated Long | 39 |
| Liquidated Cross Long | 36 |
| Liquidated Isolated Short | 1 |
| Spot Dust Conversion | 1 |

> Name: count, dtype: int64[pyarrow]

---

Fills Tagged as Liquidation-Related
-----------------------------------

| statusTier | Count |
|------------|-------|
| activeCore | 46 |
| activeLongTail | 15 |
| churnedTrader | 13 |
| silentHolder | 2 |

> dtype: int64

---

Share of Total Loss Coming from the Single Worst Trade, by Tier
---------------------------------------------------------------

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 99.0 | 0.132764 | 0.180953 | 0.010328 | 0.034875 | 0.066010 | 0.114957 | 0.912984 |
| activeLongTail | 80.0 | 0.078463 | 0.078153 | 0.003956 | 0.025820 | 0.047326 | 0.103089 | 0.414775 |
| churnedTrader | 88.0 | 0.125616 | 0.197377 | 0.003066 | 0.030540 | 0.048300 | 0.114213 | 1.000000 |
| silentHolder | 21.0 | 0.114061 | 0.216573 | 0.019299 | 0.037834 | 0.051722 | 0.075443 | 1.000000 |

---

Position Size Escalation Ratio (Last 20% Notional / First 20% Notional), by Tier
--------------------------------------------------------------------------------

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 100.0 | 4.185927 | 17.844580 | 0.061412 | 0.648868 | 1.040201 | 1.832589 | 174.605318 |
| activeLongTail | 80.0 | 1.566948 | 2.637714 | 0.151279 | 0.630691 | 1.020823 | 1.740387 | 23.310915 |
| churnedTrader | 100.0 | 1.701443 | 2.971192 | 0.064355 | 0.666402 | 1.090121 | 1.662843 | 27.877815 |
| silentHolder | 20.0 | 2.357716 | 2.872906 | 0.369117 | 0.745005 | 1.518627 | 2.330486 | 12.808838 |

---

Final Week Trading Intensity vs Overall Average, by Tier (>1 = Sped Up Before Going Quiet)
------------------------------------------------------------------------------------------

| statusTier | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|-----|-----|-----|-----|-----|-----|
| activeCore | 100.0 | 2.475545 | 4.192084 | 0.012809 | 0.424644 | 0.971255 | 2.651782 | 30.614408 |
| activeLongTail | 80.0 | 2.031254 | 6.374289 | 0.025241 | 0.230584 | 0.558923 | 1.221223 | 46.580645 |
| churnedTrader | 100.0 | 1.339681 | 2.188494 | 0.002910 | 0.204545 | 0.714286 | 1.235910 | 13.642000 |
| silentHolder | 22.0 | 2.613073 | 4.872185 | 0.054206 | 0.467877 | 1.032121 | 2.779164 | 23.330499 |

---

#### Explanation
1. Liquidation is still not the main churn mechanism

The latest output repeats the liquidation-related summary, and the core signal remains unchanged: liquidation-tagged fills are rare overall and are not concentrated in `churnedTrader`. `activeCore` has 46 liquidation-related fills, while `churnedTrader` has only 13. This means the churned group is not primarily disappearing because of forced liquidation events.

This directly weakens the earlier “single blow-up / steamroller” interpretation. If churn were mainly caused by liquidations, `churnedTrader` should be the tier with the clearest liquidation concentration, but the output shows the opposite.

2. Worst-loss concentration does not uniquely identify churned traders

The share of total loss coming from the single worst trade is close across tiers on the median: `activeCore` is 0.066, `activeLongTail` is 0.047, `churnedTrader` is 0.048, and `silentHolder` is 0.052. `churnedTrader` does have a max value of 1.000, so there are individual accounts where one trade explains all observed losses, but the median shows that this is not the typical churned-trader pattern.

The practical interpretation is that churn is not mostly explained by one catastrophic losing trade. Tail-loss cases exist, but they are outliers rather than the defining behavior of the churned tier.

3. Position-size escalation is modest for the median churned trader

The last-20%-vs-first-20% notional ratio is also not unusually high for `churnedTrader`. Its median is 1.090, only slightly above stable sizing. `activeCore` has a similar median at 1.040 but a much larger mean and max, including a max escalation ratio of 174.605.

That means the data does not support a broad “churned traders keep increasing size until they break” story. If anything, the most extreme escalation behavior appears inside `activeCore`, yet those accounts remain active. So size escalation alone cannot explain churn.

4. Final-week intensity is the strongest behavioral signal

The clearest result is final-week trading intensity. `churnedTrader` has a median final-week intensity of 0.714, below 1, meaning the typical churned account traded less in its final week than its own historical average. It also has the lowest mean final-week intensity among the four tiers.

This points toward a gradual wind-down rather than a sudden collapse. Churned traders are not generally accelerating into a final blow-up; they are slowing down before going quiet.

5. Updated interpretation: gradual disengagement, not forced exit

Putting the three mechanism checks together:
- liquidation events are rare and not concentrated in `churnedTrader`;
- worst-loss concentration is similar across tiers;
- median position escalation is modest for `churnedTrader`;
- final-week intensity is below the account's historical average.

The best current explanation is therefore **gradual disengagement**. `churnedTrader` accounts likely reduce activity over time and eventually stop, rather than being forced out by liquidation or a single dominant tail-loss event.

6. What remains unresolved

This output is strong for rejecting the blow-up hypothesis, but it does not yet explain why disengagement happens. The next analysis should focus on late-life PnL trajectory: whether win rate declines, whether average PnL per closed trade compresses, and whether the final phase contains many small losses rather than one large loss.

A useful next step is to run the lifetime-quintile PnL analysis separately and append its raw output as a new section, without modifying the existing raw tables.

### Layer 1 behavior analysis
#### Raw output
Address-level behavior summary by statusTier (median values):

| statusTier | fillCount | activeDays | activeMonths | fillsPerActiveDay | totalNotional | realizedPnl | winRate | daysSinceLastFill |
|------------|----------:|-----------:|-------------:|------------------:|--------------:|------------:|--------:|------------------:|
| activeCore | 9,962.0 | 44.5 | 6.0 | 98.843 | 21,318,460 | -1,463.670 | 0.523 | 87.0 |
| activeLongTail | 9,982.0 | 38.0 | 3.0 | 147.595 | 10,052,360 | 207.477 | 0.507 | 36.5 |
| churnedTrader | 8,589.5 | 9.0 | 2.0 | 521.908 | 49,878,530 | 37,432.771 | 0.531 | 296.0 |
| silentHolder | 7,130.0 | 71.0 | 7.0 | 76.932 | 21,388,980 | 19,658.359 | 0.468 | 129.5 |

---

Pareto concentration by statusTier:

| statusTier | top10pctNotionalShare | top20pctNotionalShare | top10pctFillShare | top20pctFillShare |
|------------|----------------------:|----------------------:|------------------:|------------------:|
| activeCore | 0.463 | 0.696 | 0.142 | 0.284 |
| activeLongTail | 0.459 | 0.710 | 0.138 | 0.275 |
| churnedTrader | 0.344 | 0.563 | 0.151 | 0.301 |
| silentHolder | 0.512 | 0.620 | 0.221 | 0.368 |

---

Fill-count funnel by statusTier (share of sampled addresses):

| statusTier | addresses | fillsGte1 | fillsGte10 | fillsGte100 | fillsGte1000 | fillsGte5000 | fillsGte10000 |
|------------|----------:|----------:|-----------:|------------:|-------------:|-------------:|--------------:|
| activeCore | 100 | 1.000 | 1.000 | 0.940 | 0.820 | 0.690 | 0.200 |
| activeLongTail | 80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.662 | 0.212 |
| churnedTrader | 100 | 1.000 | 1.000 | 0.990 | 0.930 | 0.640 | 0.280 |
| silentHolder | 22 | 1.000 | 0.909 | 0.909 | 0.909 | 0.591 | 0.091 |

---

Average cohort retention by statusTier:

| statusTier | m1Retention | m3Retention | m6Retention |
|------------|------------:|------------:|------------:|
| activeCore | 0.837 | 0.753 | 0.638 |
| activeLongTail | 0.785 | 0.522 | 0.567 |
| churnedTrader | 0.639 | 0.362 | 0.261 |
| silentHolder | 0.964 | 0.955 | 0.963 |

---

#### Explanation
1. Layer 1 confirms that churn is mostly a retention problem, not a historical-activity problem

The fill-count funnel shows `churnedTrader` accounts were not low-information accounts. In the sampled fills, 99.0% have at least 100 fills, 93.0% have at least 1,000 fills, 64.0% have at least 5,000 fills, and 28.0% hit the 10,000-fill cap. This means the churned group contains historically active traders, not random inactive wallets.

The key difference is not whether they ever traded, but whether their activity persisted. Median `daysSinceLastFill` is 296 days for `churnedTrader`, far above `activeCore` at 87 days and `activeLongTail` at 36.5 days.

2. Churned traders trade in compressed, intense bursts

`churnedTrader` has the highest median fills per active day at 521.908, compared with 98.843 for `activeCore`, 147.595 for `activeLongTail`, and 76.932 for `silentHolder`. But it has only 9 median active days and 2 median active months.

This is an important behavioral signature: churned accounts are not simply low-frequency users. They are high-intensity users over a short active window, then they disappear. That supports a “burst-and-fade” lifecycle rather than a slow, stable participation pattern.

3. ActiveCore is broader and more durable; activeLongTail is recent but narrower

`activeCore` has 44.5 median active days and 6 median active months, making it a more durable trading segment. `activeLongTail` is more recent, with only 36.5 median days since last fill, but its median activeMonths is only 3. This makes `activeLongTail` look like a currently active but less established group.

The distinction matters for product interpretation: `activeLongTail` may not be churned yet, but it may be the segment most at risk of becoming churned if its short active history fails to extend.

4. SilentHolder behaves differently from churnedTrader

`silentHolder` has the highest median activeMonths at 7 and strong cohort retention proxies, but a lower win rate. This is consistent with the earlier label: these accounts do not look like bursty churned traders. They look more like accounts with longer observed participation windows that eventually became quiet while still retaining account value.

Because only 22 `silentHolder` addresses have fills in this sample, this segment should still be interpreted cautiously.

5. Pareto concentration shows different concentration shapes by tier

The top 10% of `activeCore` addresses contribute 46.3% of notional, and the top 10% of `activeLongTail` contribute 45.9%. `silentHolder` is even more concentrated at 51.2%. By contrast, `churnedTrader` is less concentrated by notional at 34.4% for the top 10%.

This suggests the churned segment's high activity is less dependent on only a few whale-like accounts. It is more broadly distributed across the sampled churned users, which strengthens the interpretation that churn is a segment-level behavior rather than a single-outlier artifact.

6. Cohort retention is the strongest Layer 1 validation signal

Average cohort retention falls fastest for `churnedTrader`: m1 retention is 0.639, m3 retention is 0.362, and m6 retention is 0.261. `activeCore` retains much better at m6 with 0.638, while `silentHolder` remains near 0.963.

This directly validates the original Layer 1 goal: the four status tiers do correspond to different behavior patterns in the fill history. `churnedTrader` is not merely defined by current account value; it also shows materially weaker observed retention in trading activity.

### Layer 1 conclusion

Layer 1 supports the lifecycle story from the original plan:
- `activeCore`: durable, broad active history and better retention;
- `activeLongTail`: currently recent, but with a shorter observed activity history;
- `churnedTrader`: historically active, high-intensity, short-window traders with poor cohort retention;
- `silentHolder`: longer observed participation and high retention proxy, but now quiet and limited by small fill-observed sample size.

The next step is Layer 2 feature engineering and clustering. The address-level behavior table saved by this pass can be used directly as the base feature matrix, then combined with the existing `statusTier` labels to check whether unsupervised clusters rediscover the same lifecycle structure or reveal a new segmentation axis.

Note: the Layer 1 parquet tables are generated artifacts. Re-run `uv run python python/layer1BehaviorAnalysis.py` locally to recreate them; they are intentionally not tracked in git so PRs remain text-only.

Capped-address correction (supersedes the raw activeSpanDays / cohort-retention numbers above)

Why: userFillsByTime only ever returns the 10,000 most recent fills per address (server-side hard cap). For any address that hits this cap, firstFill in our data is just "earliest fill inside the most-recent-10k window," not the address's true trading start. activeSpanDays, activeMonths, and cohortMonth (and therefore cohort retention) are unreliable for these addresses. lastFill-derived metrics (daysSinceLastFill, final-week intensity, etc.) are unaffected and do not need correction.

Capped-address share by statusTier
statusTier	addresses	cappedCount	cappedShare
activeCore	100	20	0.200
activeLongTail	80	17	0.212
churnedTrader	100	28	0.280
silentHolder	22	2	0.091

(This matches the fillsGte10000 column already in the fill-count funnel table above — 67/302 addresses overall, 22.2%.)

Active span (days), full sample vs. non-capped-only
statusTier	Full-sample median	Non-capped-only median	Non-capped n
activeCore	202.5	293.5	80
activeLongTail	65.5	136.0	63
churnedTrader	14.0	15.0	72
silentHolder	226.5	245.5	20

activeCore and activeLongTail were both understated by the cap — true active windows are roughly 45% and 108% longer than the raw numbers suggested. churnedTrader's 14 → 15 day median is effectively unchanged, which independently confirms the burst-and-fade conclusion was not a capped-address artifact.

Cohort retention, corrected (cohortMonth re-derived from non-capped addresses only)
statusTier	m1Retention	m3Retention	m6Retention	Non-capped n
activeCore	0.888	0.797	0.701	80
activeLongTail	0.857	0.584	0.616	63
churnedTrader	0.660	0.494	0.388	72
silentHolder	0.962	0.955	0.963	20

Direction of the lifecycle story is unchanged (churnedTrader still retains worst, silentHolder best), but every tier's retention is higher once capped addresses are excluded from cohort assignment — the original table understated retention across the board, not just for the tiers with the largest span correction.

Caveat to carry forward: 85 of 94 non-capped cohort-month buckets contain fewer than 5 addresses. Treat the m3/m6 corrected retention numbers as directionally suggestive, not statistically precise — this is the same small-sample-decay caveat already listed in the "Known Data/Methodology Caveats" section.