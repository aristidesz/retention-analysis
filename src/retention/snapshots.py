"""Snapshot table: one row per user per prediction month.

Features use only months up to and including the snapshot month.
The label uses only the months after it. This is what keeps the model honest.
"""

import pandas as pd

from retention.config import Settings

FEATURES = [
    "active_now",
    "active_months_last_3",
    "months_since_last_active",
    "tenure",
    "card_amount_last_3",
    "card_trend",
    "p2p_months_last_3",
    "opens_now",
    "opens_last_3",
    "opens_trend",
    "searches_last_3",
    "chats_last_3",
    "chats_per_100_opens",
    "age",
    "friends",
    "android_pay_before",
    "photo_before",
    "overdraft_before",
]


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every user has a row for every month from first payment on."""
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    index = pd.MultiIndex.from_product([df["user"].unique(), months], names=["user", "month"])
    grid = df.set_index(["user", "month"]).reindex(index).reset_index()
    first = df.groupby("user")["month"].min()
    grid = grid[grid["month"] >= grid["user"].map(first)].reset_index(drop=True)
    for col in ("chats", "opens", "searches", "card", "p2p"):
        grid[col] = grid[col].fillna(0.0)
    grid["active"] = grid["active"].fillna(False).astype(bool)
    static = ["age", "friends", "android_pay", "photo", "overdraft", "first_month"]
    grid[static] = grid.groupby("user")[static].transform("first")
    return grid


def build_snapshots(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    g = complete_grid(df).sort_values(["user", "month"]).reset_index(drop=True)
    by_user = g.groupby("user")

    def rolling(col: str, window: int, how: str) -> pd.Series:
        r = by_user[col].rolling(window, min_periods=1)
        return getattr(r, how)().reset_index(level=0, drop=True)

    def previous_mean(col: str) -> pd.Series:
        return by_user[col].shift(1).groupby(g["user"]).rolling(3, min_periods=1).mean().reset_index(
            level=0, drop=True
        )

    out = pd.DataFrame({"user": g["user"], "month": g["month"]})
    out["active_now"] = g["active"].astype(int)
    out["active_months_last_3"] = rolling("active", 3, "sum")

    last_active = g["month"].where(g["active"]).groupby(g["user"]).ffill()
    out["months_since_last_active"] = [
        (m - la).n if pd.notna(la) else 99 for m, la in zip(g["month"], last_active)
    ]
    out["tenure"] = [(m - f).n for m, f in zip(g["month"], g["first_month"])]

    card_paid = g["card"].where(g["card"] != 0)
    out["card_amount_last_3"] = (
        card_paid.groupby(g["user"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    ).fillna(0.0)
    out["card_trend"] = g["card"] - previous_mean("card").fillna(0.0)
    out["p2p_months_last_3"] = (g["p2p"] != 0).astype(int).groupby(g["user"]).rolling(
        3, min_periods=1
    ).sum().reset_index(level=0, drop=True)

    out["opens_now"] = g["opens"]
    out["opens_last_3"] = rolling("opens", 3, "mean")
    out["opens_trend"] = g["opens"] - previous_mean("opens").fillna(0.0)
    out["searches_last_3"] = rolling("searches", 3, "mean")
    out["chats_last_3"] = rolling("chats", 3, "sum")
    out["chats_per_100_opens"] = 100 * out["chats_last_3"] / (3 * out["opens_last_3"]).clip(lower=1)

    out["age"] = g["age"]
    out["friends"] = g["friends"]
    month_end = g["month"].dt.end_time
    for event, name in (("android_pay", "android_pay_before"), ("photo", "photo_before"), ("overdraft", "overdraft_before")):
        out[name] = (g[event].notna() & (g[event] <= month_end)).astype(int)

    # Label: no payment in any of the next `label_months` months
    k = settings.label_months
    future = pd.concat([by_user["active"].shift(-i) for i in range(1, k + 1)], axis=1)
    out["label_known"] = future.notna().all(axis=1)
    out["snap"] = g["month"].dt.end_time
    out["label_end"] = (g["month"] + k).dt.end_time
    out["churn"] = (~future.fillna(False).astype(bool).any(axis=1)).astype(int)

    # Score only users who have not already left and have enough history
    eligible = (
        out["label_known"]
        & (out["months_since_last_active"] <= 1)
        & (out["tenure"] >= settings.min_history_months - 1)
    )
    return out[eligible].drop(columns="label_known").reset_index(drop=True)
