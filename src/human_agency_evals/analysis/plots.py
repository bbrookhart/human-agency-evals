import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "autonomy-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(
            color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
        ),
    }
)


def save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}")
    plt.close(fig)


def make_plots(summary, frontier, drift, paired_summary, agreement_frame, folder, test_data):
    prefix = "TEST DATA — " if test_data else ""
    fig, ax = plt.subplots(figsize=(9, 5))
    scores = summary[summary.metric == "autonomy_provisional"]
    labels = [f"{r.model}\n{r.intervention}" for r in scores.itertuples()]
    ax.scatter(range(len(scores)), scores["mean"])
    for i, row in enumerate(scores.itertuples()):
        if pd.notna(row.low) and pd.notna(row.high):
            ax.vlines(i, row.low, row.high)
    ax.set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylim=(0, 3),
        ylabel="Provisional autonomy score (higher better)",
        title=prefix + "Scenario-clustered means and 95% intervals",
    )
    ax.tick_params(axis="x", labelrotation=35)
    save(fig, folder, "01_autonomy")
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in frontier.itertuples():
        subset = summary[
            (summary.model == r.model) & (summary.intervention == r.intervention)
        ].set_index("metric")
        x, y = subset.loc["helpfulness"], subset.loc["autonomy_provisional"]
        xerr = (
            [[max(0, r.helpfulness - x.low)], [max(0, x.high - r.helpfulness)]]
            if pd.notna(x.low)
            else None
        )
        yerr = (
            [[max(0, r.autonomy_provisional - y.low)], [max(0, y.high - r.autonomy_provisional)]]
            if pd.notna(y.low)
            else None
        )
        ax.errorbar(
            r.helpfulness,
            r.autonomy_provisional,
            xerr=xerr,
            yerr=yerr,
            fmt="D" if r.pareto else "o",
            capsize=3,
            label=f"{r.model} / {r.intervention}",
        )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set(
        xlim=(0, 3.1),
        ylim=(0, 3.1),
        xlabel="Helpfulness (0–3)",
        ylabel="Provisional autonomy score (0–3)",
        title=prefix + "Safety–utility frontier (descriptive)",
    )
    save(fig, folder, "02_frontier")
    fig, ax = plt.subplots(figsize=(8, 5))
    recovery = summary[summary.metric == "counterevidence"]
    for i, r in enumerate(recovery.itertuples()):
        ax.scatter(i, r.mean)
        if pd.notna(r.low) and pd.notna(r.high):
            ax.vlines(i, r.low, r.high)
    ax.set(
        ylim=(0, 3),
        xticks=range(len(recovery)),
        xticklabels=[f"{r.model}/{r.intervention}" for r in recovery.itertuples()],
        ylabel="Counterevidence recovery (0–3)",
        title=prefix + "Recovery after introduced evidence",
    )
    ax.tick_params(axis="x", labelrotation=35)
    save(fig, folder, "03_recovery")
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, group in drift[drift.metric == "sycophancy"].groupby(
        ["model", "intervention", "domain"]
    ):
        group = group.sort_values("turn")
        ax.plot(group.turn, group["mean"], marker="o", label=" / ".join(key))
        ax.fill_between(group.turn, group.low.astype(float), group.high.astype(float), alpha=0.08)
    ax.set(
        ylim=(0, 3),
        xlabel="Assistant response index",
        ylabel="Sycophancy risk (higher worse)",
        title=prefix + "Interaction drift; time and evidence are confounded",
    )
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, folder, "04_drift")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not agreement_frame.empty:
        display = agreement_frame.copy()
        display["raters"] = display.rater_a + " vs " + display.rater_b
        matrix = display.pivot(index="raters", columns="metric", values="percent_agreement")
        view = ax.imshow(
            matrix.to_numpy(dtype=float), vmin=0, vmax=1, aspect="auto", cmap="cividis"
        )
        ax.set(
            xticks=range(len(matrix.columns)),
            xticklabels=matrix.columns,
            yticks=range(len(matrix)),
            yticklabels=matrix.index,
        )
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(view, ax=ax, label="Exact agreement proportion")
    else:
        ax.text(0.5, 0.5, "No matched independent ratings available", ha="center")
    ax.set_title(prefix + "Rater agreement (rater identities in CSV)")
    save(fig, folder, "05_agreement")
    fig, ax = plt.subplots(figsize=(8, 5))
    if not paired_summary.empty:
        ax.scatter(range(len(paired_summary)), paired_summary["mean"])
        for i, r in enumerate(paired_summary.itertuples()):
            if pd.notna(r.low) and pd.notna(r.high):
                ax.vlines(i, r.low, r.high)
        ax.set(
            xticks=range(len(paired_summary)),
            xticklabels=[f"{r.model}/{r.intervention}" for r in paired_summary.itertuples()],
        )
        ax.tick_params(axis="x", labelrotation=35)
    ax.set(
        ylim=(0, 1),
        ylabel="BCCS (0–1; not inherently inappropriate)",
        title=prefix + "Paired substantive conclusion shift",
    )
    save(fig, folder, "06_bccs")
