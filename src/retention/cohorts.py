"""Cohort retention and engagement by tenure (months since first payment)."""

import pandas as pd


def add_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort"] = df["first_month"]
    df["tenure"] = (df["month"] - df["cohort"]).apply(lambda offset: offset.n)
    return df


def retention_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each cohort active at each tenure month.

    Cells after the last complete month are missing, not zero, so late
    cohorts are not counted as churned before they had the chance to stay.
    """
    df = add_tenure(df)
    size = df[df["tenure"] == 0].groupby("cohort")["user"].nunique()
    active = df[df["active"]].groupby(["cohort", "tenure"])["user"].nunique().unstack()
    observed = df.groupby(["cohort", "tenure"])["user"].size().unstack().notna()
    matrix = active.div(size, axis=0).where(observed)
    matrix.insert(0, "cohort_size", size)
    return matrix


def engagement_by_tenure(df: pd.DataFrame, min_users: int = 0) -> pd.DataFrame:
    """Per active user, by tenure: compares users at the same age.

    A plain daily average over everyone mixes new and old users. When the
    user base grows fast, that average moves even if nobody changed behaviour.
    Tenures with fewer than `min_users` active users are dropped: a handful of
    long-lived early users (testers, staff) would make the averages jump.
    """
    df = add_tenure(df)
    active = df[df["active"]]
    out = active.groupby("tenure").agg(
        active_users=("user", "nunique"),
        opens_per_user=("opens", "mean"),
        searches_per_user=("searches", "mean"),
        chats_per_user=("chats", "mean"),
        share_using_p2p=("p2p", lambda s: (s != 0).mean()),
        median_card_amount=("card", "median"),
    )
    out["chats_per_100_opens"] = 100 * out["chats_per_user"] / out["opens_per_user"]
    return out[out["active_users"] >= min_users]


def chats_and_churn(df: pd.DataFrame, label_months: int) -> pd.DataFrame:
    """Do users who contact support churn more? Churn rate by chat band."""
    df = add_tenure(df).sort_values(["user", "month"])
    future = [df.groupby("user")["active"].shift(-k) for k in range(1, label_months + 1)]
    known = pd.concat(future, axis=1).notna().all(axis=1)
    churned = ~pd.concat(future, axis=1).fillna(False).astype(bool).any(axis=1)
    base = df[df["active"] & known].assign(churned=churned[df["active"] & known])
    bands = pd.cut(base["chats"], [-0.1, 0, 1, 3, float("inf")], labels=["0", "1", "2-3", "4+"])
    return base.groupby(bands, observed=True)["churned"].agg(["mean", "size"]).rename(
        columns={"mean": "churn_rate", "size": "user_months"}
    )
