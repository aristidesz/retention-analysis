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
