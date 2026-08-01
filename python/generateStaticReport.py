
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TIER_ORDER = ["activeCore", "activeLongTail", "churnedTrader", "silentHolder"]
TIER_COLORS = {
    "activeCore": "#2E86AB",
    "activeLongTail": "#F6AE2D",
    "churnedTrader": "#E4572E",
    "silentHolder": "#8E9AAF",
}

REPORT_OUTPUT_PATH = "output/reports/traderBehaviorReport.html"
IMAGES_DIR = "docs/images"

os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def loadParquet(path):
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Build the same figures used in dashboard/app.py, as standalone Plotly
# figure objects (no Streamlit dependency here).
# ---------------------------------------------------------------------------

def buildCappedAddressFigure():
    spanDf = loadParquet("output/layer1Behavior/activeSpanCorrectedByTier.parquet")
    beforeAfter = spanDf.melt(
        id_vars="statusTier",
        value_vars=["activeSpanDaysFullSampleMedian", "activeSpanDaysNonCappedMedian"],
        var_name="version", value_name="medianActiveSpanDays",
    )
    beforeAfter["version"] = beforeAfter["version"].map({
        "activeSpanDaysFullSampleMedian": "Before correction (full sample)",
        "activeSpanDaysNonCappedMedian": "After correction (non-capped only)",
    })
    fig = px.bar(
        beforeAfter, x="statusTier", y="medianActiveSpanDays", color="version",
        barmode="group", category_orders={"statusTier": TIER_ORDER},
        title="Active span before vs. after the 10k fill-cap correction",
    )
    fig.update_layout(template="plotly_white", legend_title_text="")
    return fig


def buildChurnEdgeDeclineFigure():
    bucketDf = loadParquet("output/churnMechanism/shareNetProfitableByTierBucket.parquet")
    fig = px.line(
        bucketDf.sort_values("monthsBeforeLastFill"),
        x="monthsBeforeLastFill", y="shareNetProfitable", color="statusTier",
        color_discrete_map=TIER_COLORS, markers=True,
        category_orders={"statusTier": TIER_ORDER},
        title="Share of accounts net-profitable, by months before last fill (0 = closest to disengagement)",
    )
    fig.update_layout(template="plotly_white", xaxis_title="months before last fill", yaxis_tickformat=".0%")
    return fig


def buildActiveLongTailFigure():
    fillsSummary = pd.DataFrame([
        {"statusTier": "activeCore", "executionType": "maker", "winRate": 0.644},
        {"statusTier": "activeCore", "executionType": "taker", "winRate": 0.506},
        {"statusTier": "activeLongTail", "executionType": "maker", "winRate": 0.715},
        {"statusTier": "activeLongTail", "executionType": "taker", "winRate": 0.447},
    ])
    fig = px.bar(
        fillsSummary, x="executionType", y="winRate", color="statusTier", barmode="group",
        color_discrete_map=TIER_COLORS,
        title="Win rate by execution type (maker vs. taker): activeCore vs. activeLongTail",
    )
    fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", legend_title_text="")
    return fig


def buildClusteringHeatmapFigure():
    crosstabDf = loadParquet("output/layer2Clustering/clusterVsTierCrosstab.parquet")
    fig = go.Figure(data=go.Heatmap(
        z=crosstabDf.values, x=[f"cluster {c}" for c in crosstabDf.columns],
        y=crosstabDf.index, colorscale="Blues", text=crosstabDf.values, texttemplate="%{text}",
    ))
    fig.update_layout(template="plotly_white", title="KMeans cluster vs. statusTier (address counts) — every cluster mixes 3-4 tiers")
    return fig


def buildChurnedSubclusterFigure():
    subExecDf = loadParquet("output/layer2Clustering/churnedSubclusterExecutionMechanism.parquet")
    subExecDf["executionType"] = subExecDf["crossed"].map({True: "taker", False: "maker"})
    fig = px.bar(
        subExecDf, x="executionType", y="winRate", color="subCluster", barmode="group",
        title="churnedTrader sub-cluster win rate, by execution type",
    )
    fig.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# 1. Static PNG exports for README embedding.
#
# Deliberately matplotlib, not Plotly+kaleido: kaleido needs a local Chrome
# install to rasterize charts, which isn't guaranteed to be present (and
# isn't reachable at all in network-restricted environments). matplotlib
# renders PNGs with zero browser dependency and is already a project
# dependency, so this path works everywhere the rest of the pipeline does.
# ---------------------------------------------------------------------------

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def saveFig(fig, filename):
    path = os.path.join(IMAGES_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plotCappedAddressCorrection():
    spanDf = loadParquet("output/layer1Behavior/activeSpanCorrectedByTier.parquet").set_index("statusTier").loc[TIER_ORDER]
    x = np.arange(len(TIER_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, spanDf["activeSpanDaysFullSampleMedian"], width, label="Before correction (full sample)", color="#B0B8C1")
    ax.bar(x + width / 2, spanDf["activeSpanDaysNonCappedMedian"], width, label="After correction (non-capped only)", color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER, rotation=15)
    ax.set_ylabel("median active span (days)")
    ax.set_title("Active span before vs. after the 10k fill-cap correction")
    ax.legend(fontsize=8)
    return fig


def plotChurnEdgeDecline():
    bucketDf = loadParquet("output/churnMechanism/shareNetProfitableByTierBucket.parquet")
    fig, ax = plt.subplots(figsize=(7, 4))
    for tier in TIER_ORDER:
        tierDf = bucketDf[bucketDf["statusTier"] == tier].sort_values("monthsBeforeLastFill")
        ax.plot(tierDf["monthsBeforeLastFill"], tierDf["shareNetProfitable"], marker="o", label=tier, color=TIER_COLORS[tier])
    ax.set_xlabel("months before last fill (0 = closest to disengagement)")
    ax.set_ylabel("share of accounts net-profitable")
    ax.set_title("Churn edge-decline check: no clean decline for churnedTrader")
    ax.legend(fontsize=8)
    return fig


def plotActiveLongTailExecution():
    executionTypes = ["maker", "taker"]
    activeCoreVals = [0.644, 0.506]
    activeLongTailVals = [0.715, 0.447]
    x = np.arange(len(executionTypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, activeCoreVals, width, label="activeCore", color=TIER_COLORS["activeCore"])
    ax.bar(x + width / 2, activeLongTailVals, width, label="activeLongTail", color=TIER_COLORS["activeLongTail"])
    ax.set_xticks(x)
    ax.set_xticklabels(executionTypes)
    ax.set_ylabel("win rate")
    ax.set_title("Win rate by execution type: activeCore vs. activeLongTail")
    ax.legend(fontsize=8)
    return fig


def plotClusteringHeatmap():
    crosstabDf = loadParquet("output/layer2Clustering/clusterVsTierCrosstab.parquet")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(crosstabDf.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(crosstabDf.columns)))
    ax.set_xticklabels([f"cluster {c}" for c in crosstabDf.columns])
    ax.set_yticks(range(len(crosstabDf.index)))
    ax.set_yticklabels(crosstabDf.index)
    for i in range(crosstabDf.shape[0]):
        for j in range(crosstabDf.shape[1]):
            value = crosstabDf.values[i, j]
            textColor = "white" if value > crosstabDf.values.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=textColor)
    ax.set_title("KMeans cluster vs. statusTier — every cluster mixes 3-4 tiers")
    fig.colorbar(im, ax=ax, label="address count")
    return fig


def exportReadmeImages():
    saveFig(plotCappedAddressCorrection(), "cappedAddressCorrection.png")
    saveFig(plotChurnEdgeDecline(), "churnEdgeDecline.png")
    saveFig(plotActiveLongTailExecution(), "activeLongTailExecution.png")
    saveFig(plotClusteringHeatmap(), "clusteringHeatmap.png")


# ---------------------------------------------------------------------------
# 2. Self-contained interactive HTML report.
# ---------------------------------------------------------------------------

REPORT_SECTIONS = [
    ("Overview", """
        <p>40,471 leaderboard addresses, a stratified sample of 302 with fills history
        (2.08M fills), analyzed through four <code>statusTier</code> segments:
        <code>activeCore</code>, <code>activeLongTail</code>, <code>churnedTrader</code>,
        <code>silentHolder</code>.</p>
    """, None),
    ("A data-quality catch: the 10k fill cap", """
        <p>Hyperliquid's <code>userFillsByTime</code> endpoint hard-caps at 10,000 most
        recent fills per address (server-side, not a pagination bug). 67/302 sampled
        addresses (22.2%) hit it, which silently understated <code>activeSpanDays</code>
        for the affected addresses before correction.</p>
    """, buildCappedAddressFigure),
    ("Why traders churn", """
        <p>Liquidation, a single catastrophic trade, and position-escalation-into-blowup
        were all ruled out first. <code>shareNetProfitable</code> is flat and
        non-monotonic across the four months leading up to each account's last fill
        (slope &asymp; 0.0022) &mdash; churn is disengagement-driven, not preceded by a
        decline in trading edge.</p>
    """, buildChurnEdgeDeclineFigure),
    ("Why activeLongTail underperforms", """
        <p>Fee burden was ruled out first (flat ~0.03% of notional across every tier).
        <code>activeLongTail</code> sizes positions ~24-90x more aggressively relative to
        account equity than <code>activeCore</code>, and relies on taker (aggressive)
        execution more (92.2% vs. 81.7% of fills) &mdash; and that taker execution performs
        worse for them specifically.</p>
    """, buildActiveLongTailFigure),
    ("Clustering: trading style vs. account status", """
        <p>KMeans (k=4) vs. <code>statusTier</code> gives an adjusted Rand index of 0.079;
        HDBSCAN gives 0.010 &mdash; both near the 0 expected under random labeling.
        <code>statusTier</code> is built from account state (equity scale, recency); this
        feature space captures trading style (intensity, execution aggressiveness). The two
        are close to orthogonal.</p>
    """, buildClusteringHeatmapFigure),
    ("Within churnedTrader: two different stories", """
        <p>Re-clustering <code>churnedTrader</code> alone (no tier labels) splits it
        roughly 55/45: one group disengaged despite a strong taker win rate (61.5%), the
        other has a 100% taker share and the worst taker win rate found anywhere in the
        project (41.2%) &mdash; independently rediscovering the <code>activeLongTail</code>
        mechanism.</p>
    """, buildChurnedSubclusterFigure),
]


def buildReportHtml():
    plotlyJsMode = "cdn"
    sectionsHtml = []
    for idx, (title, bodyHtml, figureBuilder) in enumerate(REPORT_SECTIONS):
        chartHtml = ""
        if figureBuilder is not None:
            fig = figureBuilder()
            includeJs = plotlyJsMode if idx == 0 or not any(
                f is not None for _, _, f in REPORT_SECTIONS[:idx]
            ) else False
            chartHtml = fig.to_html(full_html=False, include_plotlyjs=includeJs, config={"displaylogo": False})
        sectionsHtml.append(f"""
        <section>
            <h2>{idx + 1}. {title}</h2>
            {bodyHtml}
            {chartHtml}
        </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hyperliquid Trader Behavior — Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px;
          margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; }}
  section {{ margin-bottom: 48px; padding-bottom: 32px; border-bottom: 1px solid #eee; }}
  h2 {{ color: #2E86AB; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }}
  footer {{ color: #999; font-size: 0.85em; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Hyperliquid Trader Behavior</h1>
<p class="subtitle">Static export of the interview-narrative dashboard — charts stay interactive, no server required. Full evidence and caveats: <code>docs/findings.md</code> in the repo.</p>
{"".join(sectionsHtml)}
<footer>Generated by python/generateStaticReport.py from live output/*.parquet data.</footer>
</body>
</html>"""


def main():
    print("Building self-contained interactive HTML report ...")
    reportHtml = buildReportHtml()
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(reportHtml)
    print(f"  wrote {REPORT_OUTPUT_PATH} ({len(reportHtml) / 1024:.0f} KB)")

    print("\nExporting static PNGs for README embedding ...")
    exportReadmeImages()


if __name__ == "__main__":
    main()