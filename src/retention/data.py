"""Load the monthly core table and define who is active."""

import pandas as pd

from retention.config import Columns, Settings

NUMERIC = ("chats", "opens", "searches", "card", "p2p", "age", "friends")
EVENTS = ("android_pay", "photo", "overdraft")


def load_core(settings: Settings) -> pd.DataFrame:
    """Read the core table, check the columns and return one row per user-month."""
    c = settings.columns
    wanted = [getattr(c, name) for name in ("user", "year", "month") + NUMERIC + EVENTS]
    header = pd.read_csv(settings.core_path, nrows=0).columns
    missing = [col for col in wanted if col not in header]
    if missing:
        raise ValueError(
            f"Columns not found in {settings.core_path}: {missing}. "
            f"Available: {list(header)}. Update Columns in config.py."
        )
    raw = pd.read_csv(settings.core_path, usecols=wanted, low_memory=False)
    return prepare(raw, c)


def prepare(raw: pd.DataFrame, c: Columns) -> pd.DataFrame:
    """Rename to short names, build the month period and the active flag."""
    rename = {getattr(c, name): name for name in ("user",) + NUMERIC + EVENTS}
    df = raw.rename(columns=rename)
    df["month"] = pd.PeriodIndex.from_fields(
        year=raw[c.year].astype(int), month=raw[c.month].astype(int), freq="M"
    )
    df = df.drop(columns=[c.year, c.month])

    for col in ("chats", "opens", "searches", "card", "p2p"):
        df[col] = df[col].fillna(0.0)
    for col in EVENTS:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    # Active = at least one payment in the month
    df["active"] = (df["card"] != 0) | (df["p2p"] != 0)

    df = df.drop_duplicates(["user", "month"]).sort_values(["user", "month"])
    return df.reset_index(drop=True)


def restrict_to_observed(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Keep months from each user's first payment up to the last complete month."""
    last = pd.Period(settings.last_complete_month, freq="M")
    first_active = df[df["active"]].groupby("user")["month"].min().rename("first_month")
    df = df.merge(first_active, on="user", how="inner")
    keep = (df["month"] >= df["first_month"]) & (df["month"] <= last)
    return df[keep].reset_index(drop=True)
