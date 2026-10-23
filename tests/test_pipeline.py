import pandas as pd
import pytest

from retention import cohorts, data, model, snapshots
from retention.config import Columns, Settings
from synthetic import make_core

SETTINGS = Settings(min_history_months=1)


@pytest.fixture(scope="module")
def df():
    return data.restrict_to_observed(data.prepare(make_core(), Columns()), SETTINGS)


@pytest.fixture(scope="module")
def snaps(df):
    return snapshots.build_snapshots(df, SETTINGS)


def test_active_means_a_payment(df):
    assert (df.loc[df["active"], ["card", "p2p"]].ne(0).any(axis=1)).all()


def test_partial_month_dropped(df):
    assert df["month"].max() == pd.Period(SETTINGS.last_complete_month, "M")


def test_retention_starts_at_100pct_and_late_cohorts_are_missing_not_zero(df):
    m = cohorts.retention_matrix(df)
    assert (m[0] == 1).all()
    last = df["month"].max()
    for cohort, row in m.drop(columns="cohort_size").iterrows():
        observable = (last - cohort).n
        assert row[row.index > observable].isna().all()
        assert row[row.index <= observable].notna().all()


def test_label_is_future_activity(df, snaps):
    active = df.set_index(["user", "month"])["active"]
    for _, row in snaps.sample(50, random_state=0).iterrows():
        nxt = [active.get((row["user"], row["month"] + k), False) for k in (1, 2)]
        assert row["churn"] == int(not any(nxt))


def test_features_do_not_change_when_the_future_changes(df):
    """Leakage test: wipe all activity after a cut-off, features before it must not move."""
    cut = pd.Period("2018-02", "M")
    before = snapshots.build_snapshots(df, SETTINGS)
    future_wiped = df.copy()
    late = future_wiped["month"] > cut
    future_wiped.loc[late, ["chats", "opens", "searches", "card", "p2p"]] = 0
    future_wiped.loc[late, "active"] = False
    after = snapshots.build_snapshots(future_wiped, Settings(min_history_months=1, last_complete_month="2018-05"))
    keys = ["user", "month"]
    a = before[before["month"] <= cut - 2].set_index(keys)[snapshots.FEATURES]
    b = after.set_index(keys)[snapshots.FEATURES].reindex(a.index)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_time_split_has_gap(snaps):
    split = model.time_split(snaps, SETTINGS)
    assert split.train["month"].max() + SETTINGS.label_months <= split.test_month


def test_run_produces_metrics(snaps):
    results = model.run(snaps, SETTINGS)
    assert {"rule_baseline", "logistic", "gradient_boosting"} <= set(results["models"])
    for scores in results["models"].values():
        assert 0 <= scores["pr_auc"] <= 1
