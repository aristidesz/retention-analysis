from dataclasses import replace

import pandas as pd
import pytest

from retention import daily, model
from retention.config import Settings
from synthetic import make_daily


def build(tmp_path, tx, act, users) -> pd.DataFrame:
    paths = {}
    for name, frame in (("tx", tx), ("act", act), ("users", users)):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    settings = Settings(
        input="daily",
        transactions_path=str(paths["tx"]),
        daily_path=str(paths["act"]),
        users_path=str(paths["users"]),
    )
    return daily.build_snapshots(settings)


@pytest.fixture(scope="module")
def tables():
    return make_daily()


@pytest.fixture(scope="module")
def snaps(tables, tmp_path_factory):
    return build(tmp_path_factory.mktemp("daily"), *tables)


def test_snapshots_are_mondays_and_ghost_is_never_scored(snaps):
    assert (snaps["snap"].dt.day_of_week == 0).all()
    assert "ghost" not in set(snaps["user"])  # +-0.01 p2p and refunds are not payments


def test_label_is_no_payment_in_next_60_days(snaps, tables):
    tx = tables[0]
    tx = tx[((tx["payment_type"] == "mastercard") & ~(tx["amount"] >= 0)) | (tx["amount"].abs() >= 1)]
    tx = tx.assign(ts=pd.to_datetime(tx["timestamp"], utc=True).dt.tz_localize(None))
    for _, row in snaps.sample(50, random_state=0).iterrows():
        mine = tx[tx["user_id"] == row["user"]]["ts"]
        paid_next = ((mine >= row["snap"]) & (mine < row["snap"] + pd.Timedelta(days=60))).any()
        paid_before = ((mine >= row["snap"] - pd.Timedelta(days=60)) & (mine < row["snap"])).any()
        assert row["churn"] == int(not paid_next)
        assert paid_before


def test_features_do_not_change_when_the_future_changes(snaps, tables, tmp_path):
    """Wipe all activity after a cut-off date: features of earlier snapshots must not move."""
    tx, act, users = tables
    cut = pd.Timestamp("2018-03-12")
    tx_ts = pd.to_datetime(tx["timestamp"], utc=True).dt.tz_localize(None)
    keep_tx = tx[tx_ts < cut]
    anchor = tx[tx_ts == tx_ts.max()].iloc[:1].assign(user_id="anchor")  # keep the data end fixed
    act_ts = pd.to_datetime(act["date"], utc=True).dt.tz_localize(None)
    wiped = build(tmp_path, pd.concat([keep_tx, anchor]), act[act_ts < cut], users)
    keys = ["user", "snap"]
    a = snaps[snaps["snap"] <= cut].set_index(keys)[daily.FEATURES]
    b = wiped.set_index(keys)[daily.FEATURES].reindex(a.index)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_daily_backtest_keeps_the_gap(snaps):
    settings = replace(Settings(), backtest_min_train_rows=1, test_window_days=7)
    split = model.time_split(snaps, settings)
    assert split.train["label_end"].max() <= split.test_start
    folds = model.backtest_folds(snaps, settings)
    assert len(folds) > 1
    for fold in folds:
        assert fold.train["label_end"].max() <= fold.test_start
