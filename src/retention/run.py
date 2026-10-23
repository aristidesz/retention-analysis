"""Run the whole analysis: python -m retention.run"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from retention import cohorts, daily, data, model, snapshots  # noqa: E402
from retention.config import Settings  # noqa: E402


def plot_retention(matrix, path: Path) -> None:
    values = matrix.drop(columns="cohort_size")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(values.to_numpy(dtype=float), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(values.shape[1]), values.columns)
    ax.set_yticks(range(values.shape[0]), [f"{c} (n={n})" for c, n in matrix["cohort_size"].items()])
    ax.set_xlabel("Months since first payment")
    ax.set_title("Share of each cohort still paying")
    for (i, j), v in np.ndenumerate(values.to_numpy(dtype=float)):
        if v == v:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_engagement(table, path: Path) -> None:
    cols = ["opens_per_user", "searches_per_user", "chats_per_100_opens"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, cols):
        ax.plot(table.index, table[col], marker="o")
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Months since first payment")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", default=None, help="Path to the monthly core CSV")
    args = parser.parse_args()
    settings = Settings() if args.core is None else Settings(core_path=args.core)

    out = Path(settings.reports_dir)
    out.mkdir(exist_ok=True)

    df = data.restrict_to_observed(data.load_core(settings), settings)
    print(f"Users with at least one payment: {df['user'].nunique():,}")

    matrix = cohorts.retention_matrix(df)
    matrix.to_csv(out / "retention_matrix.csv")
    plot_retention(matrix, out / "retention_matrix.png")

    engagement = cohorts.engagement_by_tenure(df)
    engagement.to_csv(out / "engagement_by_tenure.csv")
    plot_engagement(engagement, out / "engagement_by_tenure.png")

    chats = cohorts.chats_and_churn(df, settings.label_months)
    chats.to_csv(out / "chats_and_churn.csv")

    snaps = daily.build_snapshots(settings) if settings.input == "daily" else snapshots.build_snapshots(df, settings)
    print(f"Snapshots ({settings.input}): {len(snaps):,} rows, {snaps['snap'].min().date()} to {snaps['snap'].max().date()}")
    results = model.run(snaps, settings)

    fitted, split = results.pop("_fitted"), results.pop("_split")

    bt = model.backtest(snaps, settings)
    bt.to_csv(out / "backtest.csv", index=False)
    results["backtest"] = model.summarize_backtest(bt)
    print(bt.round(3).to_string(index=False))
    print(bt.drop(columns=["test_start", "test_end"]).groupby("model").mean(numeric_only=True).round(3).to_string())
    try:
        import shap

        explainer = shap.Explainer(fitted.predict_proba, split.train[model.feature_columns(split.train)].sample(100, random_state=0))
        values = explainer(split.test[model.feature_columns(split.test)].sample(min(300, len(split.test)), random_state=0))
        shap.plots.beeswarm(values[:, :, 1], show=False)
        plt.tight_layout()
        plt.savefig(out / "shap_beeswarm.png", dpi=120)
        plt.close()
    except ImportError:
        print("shap not installed: skipped the SHAP plot (pip install '.[shap]')")

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
