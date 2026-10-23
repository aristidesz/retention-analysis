"""Time-based split, baselines, models and business threshold."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retention.config import Settings
from retention.snapshots import FEATURES


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_month: pd.Period


def time_split(snapshots: pd.DataFrame, settings: Settings) -> Split:
    """Test on the last snapshot month. Train only on snapshots whose label
    window had already ended by then, so no future outcome is used."""
    test_month = snapshots["month"].max()
    train_end = test_month - settings.label_months
    train = snapshots[snapshots["month"] <= train_end]
    test = snapshots[snapshots["month"] == test_month]
    if train.empty:
        raise ValueError("No training snapshots before the gap. Lower min_history_months.")
    return Split(train, test, test_month)


def backtest_folds(snapshots: pd.DataFrame, settings: Settings):
    """One Split per candidate test month, each with the same gap as time_split."""
    for month in sorted(snapshots["month"].unique()):
        train = snapshots[snapshots["month"] <= month - settings.label_months]
        if len(train) >= settings.backtest_min_train_rows:
            yield Split(train, snapshots[snapshots["month"] == month], month)


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """What a team would do without a model: inactive this month = at risk."""
    return (1 - df["active_now"]).to_numpy() * 0.5 + (3 - df["active_months_last_3"]).to_numpy() / 6


def make_models(settings: Settings) -> dict:
    # No class weights: churn is not rare here, and unweighted models keep
    # probabilities usable for the cost-based threshold.
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=settings.random_state
        ),
    }


def top_share_metrics(y: np.ndarray, score: np.ndarray, share: float) -> tuple[float, float]:
    """Recall and precision if the team contacts the top `share` of users."""
    n = max(1, int(round(share * len(y))))
    top = np.argsort(-score)[:n]
    caught = y[top].sum()
    return caught / max(1, y.sum()), caught / n


def evaluate(y: np.ndarray, score: np.ndarray, settings: Settings) -> dict:
    recall_top, precision_top = top_share_metrics(y, score, settings.top_share)
    return {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        f"recall_top_{int(settings.top_share * 100)}pct": recall_top,
        f"precision_top_{int(settings.top_share * 100)}pct": precision_top,
        f"lift_top_{int(settings.top_share * 100)}pct": precision_top / y.mean(),
    }


def break_even_probability(settings: Settings) -> float:
    """Contact when p x success rate x value > offer cost."""
    return settings.offer_cost / (settings.offer_success_rate * settings.customer_value)


def expected_gain(y: np.ndarray, p: np.ndarray, settings: Settings) -> float:
    """Gain on the test month from contacting everyone above break-even."""
    contact = p >= break_even_probability(settings)
    saved = y[contact].sum() * settings.offer_success_rate * settings.customer_value
    return saved - contact.sum() * settings.offer_cost


def backtest(snapshots: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Metrics per model for every test month that has enough training data."""
    rows = []
    for split in backtest_folds(snapshots, settings):
        X_train, y_train = split.train[FEATURES], split.train["churn"].to_numpy()
        X_test, y_test = split.test[FEATURES], split.test["churn"].to_numpy()
        if y_test.min() == y_test.max():
            continue
        scores = {"rule_baseline": rule_baseline(split.test)}
        for name, model in make_models(settings).items():
            scores[name] = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        for name, score in scores.items():
            rows.append(
                {
                    "test_month": str(split.test_month),
                    "train_rows": len(split.train),
                    "test_rows": len(split.test),
                    "churn_rate": y_test.mean(),
                    "model": name,
                    **evaluate(y_test, score, settings),
                }
            )
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> dict:
    """Mean and range of each metric per model across the test months."""
    metrics = [c for c in bt.columns if c.startswith(("pr_auc", "roc_auc", "recall_top", "precision_top", "lift_top"))]
    summary = bt.groupby("model")[metrics].agg(["mean", "min", "max"])
    return {
        model: {m: {k: float(summary.loc[model, (m, k)]) for k in ("mean", "min", "max")} for m in metrics}
        for model in summary.index
    }


def run(snapshots: pd.DataFrame, settings: Settings) -> dict:
    split = time_split(snapshots, settings)
    X_train, y_train = split.train[FEATURES], split.train["churn"].to_numpy()
    X_test, y_test = split.test[FEATURES], split.test["churn"].to_numpy()

    results = {
        "test_month": str(split.test_month),
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "churn_rate_test": float(y_test.mean()),
        "break_even_probability": break_even_probability(settings),
        "models": {"rule_baseline": evaluate(y_test, rule_baseline(split.test), settings)},
    }
    fitted = {}
    for name, model in make_models(settings).items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        scores = evaluate(y_test, p, settings)
        scores["brier"] = brier_score_loss(y_test, p)
        scores["expected_gain"] = expected_gain(y_test, p, settings)
        results["models"][name] = scores
        fitted[name] = model

    best = max(fitted, key=lambda n: results["models"][n]["pr_auc"])
    imp = permutation_importance(
        fitted[best], X_test, y_test, scoring="average_precision", n_repeats=5, random_state=settings.random_state
    )
    results["best_model"] = best
    results["importance"] = (
        pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4).to_dict()
    )
    results["_fitted"] = fitted[best]
    results["_split"] = split
    return results
