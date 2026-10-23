"""Weekly snapshots built from the raw daily tables with DuckDB.

A snapshot is a Monday 00:00. Features use only events before it. The label
uses only the `label_days` days from it on. Users are scored only if they paid
in the `lookback_days` days before it.
"""

import duckdb
import pandas as pd

from retention.config import Settings

FEATURES = [
    "days_since_last_payment",
    "days_since_last_open",
    "pay_count_30d",
    "card_spend_30d",
    "p2p_out_30d",
    "p2p_in_30d",
    "active_days_30",
    "active_days_90",
    "opens_7d",
    "opens_trend",
    "searches_30d",
    "chats_30d",
    "chats_per_100_opens",
    "age",
    "friends",
    "android_pay_before",
    "photo_before",
]


def build_snapshots(settings: Settings) -> pd.DataFrame:
    return snapshot_features(load(settings), settings)


def load(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Load the raw files into DuckDB and build the intermediate tables:
    raw (every transaction), pay (payments only), mondays (snapshot dates),
    snaps (who is scored on which Monday) and act (daily app use)."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    thr, label, look = settings.min_p2p_amount, settings.label_days, settings.lookback_days

    con.execute(
        f"""
        CREATE TEMP TABLE raw AS
        SELECT user_id AS u, CAST(timestamp AS TIMESTAMP) AS ts, payment_type,
               TRY_CAST(amount AS DOUBLE) AS amount
        FROM read_csv('{settings.transactions_path}', all_varchar=true)
        """
    )
    # A payment is a card spend (refunds and zero amounts are not payments)
    # or a p2p transfer of at least `thr` pounds in either direction.
    con.execute(
        f"""
        CREATE TEMP TABLE pay AS
        SELECT u, ts, amount,
               CASE WHEN payment_type = 'mastercard' THEN 'card'
                    WHEN amount < 0 THEN 'p2p_out' ELSE 'p2p_in' END AS kind
        FROM raw
        WHERE (payment_type = 'mastercard' AND (amount IS NULL OR amount < 0))
           OR (payment_type = 'p2p_payment' AND abs(amount) >= {thr})
        """
    )
    p2p_all, p2p_kept = con.execute(
        "SELECT (SELECT count(*) FROM raw WHERE payment_type = 'p2p_payment'),"
        "       (SELECT count(*) FROM pay WHERE kind != 'card')"
    ).fetchall()[0]
    card_all, card_kept = con.execute(
        "SELECT (SELECT count(*) FROM raw WHERE payment_type = 'mastercard'),"
        "       (SELECT count(*) FROM pay WHERE kind = 'card')"
    ).fetchall()[0]
    print(f"p2p rows removed (|amount| < {thr}): {p2p_all - p2p_kept:,} of {p2p_all:,}")
    print(f"card rows removed (refunds or zero): {card_all - card_kept:,} of {card_all:,}")

    # The last day is partial, so data is complete up to its start.
    first_ts, last_ts = con.execute("SELECT min(ts), max(ts) FROM pay").fetchall()[0]
    data_end = pd.Timestamp(last_ts).normalize()
    mondays = pd.DataFrame(
        {
            "snap": pd.date_range(
                pd.Timestamp(first_ts).normalize() + pd.Timedelta(days=look),
                data_end - pd.Timedelta(days=label),
                freq="W-MON",
            )
        }
    )
    con.register("mondays", mondays)

    con.execute(
        f"""
        CREATE TEMP TABLE snaps AS
        SELECT DISTINCT p.u, m.snap FROM pay p JOIN mondays m
          ON p.ts >= m.snap - INTERVAL {look} DAY AND p.ts < m.snap
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE act AS
        SELECT user_id AS u, CAST(date AS DATE) AS d,
               TRY_CAST(app_opens AS DOUBLE) AS opens,
               TRY_CAST(app_searches AS DOUBLE) AS searches,
               TRY_CAST(chat_conversations AS DOUBLE) AS chats
        FROM read_csv('{settings.daily_path}', all_varchar=true)
        """
    )
    return con


def snapshot_features(con: duckdb.DuckDBPyConnection, settings: Settings) -> pd.DataFrame:
    """One row per user per Monday: features from before it, label from after it."""
    label = settings.label_days
    out = con.execute(
        f"""
        WITH
        paid AS (
          SELECT s.u, s.snap,
            date_diff('second', max(p.ts), s.snap) / 86400.0 AS days_since_last_payment,
            count(*) FILTER (WHERE p.ts >= s.snap - INTERVAL 30 DAY) AS pay_count_30d,
            coalesce(-sum(p.amount) FILTER (WHERE p.kind = 'card' AND p.ts >= s.snap - INTERVAL 30 DAY), 0) AS card_spend_30d,
            coalesce(-sum(p.amount) FILTER (WHERE p.kind = 'p2p_out' AND p.ts >= s.snap - INTERVAL 30 DAY), 0) AS p2p_out_30d,
            coalesce(sum(p.amount) FILTER (WHERE p.kind = 'p2p_in' AND p.ts >= s.snap - INTERVAL 30 DAY), 0) AS p2p_in_30d,
            count(DISTINCT CAST(p.ts AS DATE)) FILTER (WHERE p.ts >= s.snap - INTERVAL 30 DAY) AS active_days_30,
            count(DISTINCT CAST(p.ts AS DATE)) AS active_days_90
          FROM snaps s JOIN pay p
            ON p.u = s.u AND p.ts < s.snap AND p.ts >= s.snap - INTERVAL 90 DAY
          GROUP BY s.u, s.snap
        ),
        used AS (
          SELECT s.u, s.snap,
            coalesce(sum(a.opens) FILTER (WHERE a.d >= CAST(s.snap AS DATE) - 7), 0) AS opens_7d,
            coalesce(sum(a.opens) FILTER (WHERE a.d >= CAST(s.snap AS DATE) - 28 AND a.d < CAST(s.snap AS DATE) - 7), 0) AS opens_prev21,
            coalesce(sum(a.opens) FILTER (WHERE a.d >= CAST(s.snap AS DATE) - 30), 0) AS opens_30d,
            coalesce(sum(a.searches) FILTER (WHERE a.d >= CAST(s.snap AS DATE) - 30), 0) AS searches_30d,
            coalesce(sum(a.chats) FILTER (WHERE a.d >= CAST(s.snap AS DATE) - 30), 0) AS chats_30d,
            date_diff('day', max(a.d) FILTER (WHERE a.opens > 0), CAST(s.snap AS DATE)) AS days_since_last_open
          FROM snaps s LEFT JOIN act a
            ON a.u = s.u AND a.d < CAST(s.snap AS DATE) AND a.d >= CAST(s.snap AS DATE) - 90
          GROUP BY s.u, s.snap
        ),
        label AS (
          SELECT s.u, s.snap, count(p.ts) = 0 AS churn
          FROM snaps s LEFT JOIN pay p
            ON p.u = s.u AND p.ts >= s.snap AND p.ts < s.snap + INTERVAL {label} DAY
          GROUP BY s.u, s.snap
        ),
        profile AS (
          SELECT user_id AS u, TRY_CAST(age AS DOUBLE) AS age,
                 TRY_CAST(friends_on_monzo AS DOUBLE) AS friends,
                 TRY_CAST(android_pay_activated AS TIMESTAMP) AS android_pay,
                 TRY_CAST(profile_photo_upload AS TIMESTAMP) AS photo
          FROM read_csv('{settings.users_path}', all_varchar=true)
        )
        SELECT paid.u AS "user", paid.snap, paid.snap + INTERVAL {label} DAY AS label_end,
               paid.days_since_last_payment,
               coalesce(used.days_since_last_open, 90) AS days_since_last_open,
               paid.pay_count_30d, paid.card_spend_30d, paid.p2p_out_30d, paid.p2p_in_30d,
               paid.active_days_30, paid.active_days_90,
               used.opens_7d, used.opens_7d / 7.0 - used.opens_prev21 / 21.0 AS opens_trend,
               used.searches_30d, used.chats_30d,
               100.0 * used.chats_30d / greatest(used.opens_30d, 1) AS chats_per_100_opens,
               profile.age, profile.friends,
               CAST(profile.android_pay < paid.snap AS INTEGER) AS android_pay_before,
               CAST(profile.photo < paid.snap AS INTEGER) AS photo_before,
               CAST(label.churn AS INTEGER) AS churn
        FROM paid
        JOIN used USING (u, snap)
        JOIN label USING (u, snap)
        LEFT JOIN profile USING (u)
        ORDER BY paid.snap, paid.u
        """
    ).df()
    for col in ("android_pay_before", "photo_before"):
        out[col] = out[col].fillna(0).astype(int)
    return out
