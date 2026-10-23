"""Synthetic data with the same columns as the real core table, for tests."""

import numpy as np
import pandas as pd


def make_core(n_users: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2017-11", "2018-06", freq="M")
    rows = []
    for user in range(n_users):
        start = rng.integers(0, 5)
        stay = rng.integers(1, len(months) + 1)
        age = int(rng.integers(18, 70))
        friends = int(rng.integers(0, 40))
        pay_date = months[start].start_time + pd.Timedelta(days=int(rng.integers(0, 120)))
        for i, m in enumerate(months):
            active = start <= i < start + stay
            rows.append(
                {
                    "user_id": f"u{user}",
                    "Year": m.year,
                    "Month": m.month,
                    "chat_conversations": rng.poisson(0.3) if active else 0,
                    "app_opens": rng.poisson(30) if active else rng.poisson(1),
                    "app_searches": rng.poisson(2) if active else 0,
                    "mastercard": -rng.gamma(2, 20) if active else np.nan,
                    "p2p_payment": rng.normal(0, 30) if active and rng.random() < 0.4 else np.nan,
                    "age": age,
                    "friends_on_monzo": friends,
                    "android_pay_activated": str(pay_date) if rng.random() < 0.3 else np.nan,
                    "profile_photo_upload": np.nan,
                    "offered_overdraft": np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_daily(n_users: int = 150, seed: int = 0):
    """Synthetic transactions, daily activity and users tables, as DataFrames."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2017-11-01", "2018-06-14", freq="D", tz="UTC")
    tx, act, users = [], [], []
    for user in range(n_users):
        uid = f"u{user}"
        start = int(rng.integers(0, 120))
        stay = int(rng.integers(20, len(days)))
        users.append(
            {
                "user_id": uid,
                "account_activation": str(days[start]),
                "friends_on_monzo": int(rng.integers(0, 40)),
                "age": int(rng.integers(18, 70)),
                "profile_photo_upload": str(days[start + 2]) if rng.random() < 0.5 else np.nan,
                "android_pay_activated": np.nan,
                "offered_overdraft": np.nan,
            }
        )
        for i, day in enumerate(days):
            on = start <= i < start + stay
            act.append(
                {
                    "user_id": uid,
                    "date": str(day),
                    "chat_conversations": rng.poisson(0.1),
                    "app_opens": rng.poisson(3 if on else 0.2),
                    "app_searches": rng.poisson(0.5),
                }
            )
            if on and rng.random() < 0.3:
                ts = day + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
                kind = rng.random()
                if kind < 0.8:
                    amount = -rng.gamma(2, 8) if rng.random() < 0.95 else np.nan
                    tx.append({"timestamp": str(ts), "user_id": uid, "payment_type": "mastercard", "amount": amount})
                else:
                    tx.append({"timestamp": str(ts), "user_id": uid, "payment_type": "p2p_payment", "amount": rng.normal(0, 30)})
    # A user who only ever makes test transfers and gets refunds: never active
    for day in days[::7]:
        tx.append({"timestamp": str(day), "user_id": "ghost", "payment_type": "p2p_payment", "amount": 0.01})
        tx.append({"timestamp": str(day), "user_id": "ghost", "payment_type": "mastercard", "amount": 5.0})
    return pd.DataFrame(tx), pd.DataFrame(act), pd.DataFrame(users)
