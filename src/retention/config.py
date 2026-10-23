"""Settings for the analysis. Change them here, not inside the code."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Columns:
    """Column names in the monthly core table (data/df_core.csv)."""

    user: str = "user_id"
    year: str = "Year"
    month: str = "Month"
    chats: str = "chat_conversations"
    opens: str = "app_opens"
    searches: str = "app_searches"
    card: str = "mastercard"
    p2p: str = "p2p_payment"
    age: str = "age"
    friends: str = "friends_on_monzo"
    # Dates when the event happened, empty if it never did
    android_pay: str = "android_pay_activated"
    photo: str = "profile_photo_upload"
    overdraft: str = "offered_overdraft"


@dataclass(frozen=True)
class Settings:
    columns: Columns = field(default_factory=Columns)
    # "daily": weekly snapshots from the raw daily tables. "core": month-end
    # snapshots from the monthly core table (fallback).
    input: str = "daily"
    core_path: str = "data/df_core.csv"
    transactions_path: str = "data/df_transactions.csv"
    daily_path: str = "data/df_users_activity_daily.csv"
    users_path: str = "data/df_users.csv"
    reports_dir: str = "reports"

    # A user is active in a month if they made a payment (card or p2p).
    # App opens alone do not count: they are usage, not value.

    # Churn = no payment in the next `label_months` months after the snapshot.
    label_months: int = 2
    # Daily input: churn = no payment in the next `label_days` days after a
    # Monday snapshot; users must have paid in the `lookback_days` before it.
    label_days: int = 60
    lookback_days: int = 60
    # p2p transfers below this many pounds are treated as test transfers and
    # ignored (card amounts are real pounds; p2p has many +-0.01 rows).
    min_p2p_amount: float = 1.0
    # Backtest and final test use windows of this many days.
    test_window_days: int = 28

    # Months of history needed before a snapshot can be scored.
    min_history_months: int = 2
    # Last fully observed month. June 2018 is partial in the source data.
    last_complete_month: str = "2018-05"

    # Backtest: skip test months with fewer training rows than this.
    backtest_min_train_rows: int = 2000

    # Share of users a retention team can contact, for recall at the top.
    top_share: float = 0.10

    # Business inputs for the contact threshold (placeholders, set with the business).
    offer_cost: float = 20.0
    offer_success_rate: float = 0.20
    customer_value: float = 300.0

    random_state: int = 0
