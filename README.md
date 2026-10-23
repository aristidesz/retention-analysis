# Neobank retention analysis

Who stays, who leaves, and who to contact before they go. Built on a neobank's app and payment data (about 57k users, Nov 2017 to Jun 2018).

The data is not in this repo. Put the exported tables in `data/` (git-ignored) and run the pipeline locally.

## Questions

1. How should engagement and retention be measured?
2. How does retention look by cohort, and where are most users lost?
3. Which users are likely to stop, early enough to act?
4. Who should the retention team contact, given what an offer costs?

## Approach

**Active means paying.** A user is active in a month if they made a card or p2p payment. App opens are usage, not value, so they do not count. Chats are kept apart as a friction signal, not added to engagement.

**Engagement is a set of measures, not one score.** Frequency (active months), recency (months since last payment), depth (amounts), breadth (card and p2p), app use (opens, searches) and friction (chats per 100 opens). Adding them together would mix units and hide what moves.

**Cohorts by tenure.** Users are grouped by the month of their first payment and compared at the same age (months since first payment). Months not yet observed are left empty, not counted as churn, so late cohorts are not penalised.

**Engagement per active user by tenure.** The user base grew fast, so a plain average over everyone moves even when nobody changes behaviour. Comparing users at the same tenure removes that mix effect.

**Input.** `Settings.input` picks the source. `daily` (default) builds weekly snapshots from the raw tables with DuckDB: `df_transactions.csv`, `df_users_activity_daily.csv`, `df_users.csv`. `core` is the fallback: month-end snapshots from `df_core.csv`. Cohorts and engagement by tenure always use `df_core.csv`.

**What counts as a payment (daily input).** A card spend (refunds and zero amounts are not payments) or a p2p transfer of at least £1 in either direction (`min_p2p_amount`). Card amounts are real pounds; p2p has thousands of exactly +-0.01 rows, treated as test transfers and ignored. The pipeline prints how many rows this removes.

**Churn label.** Daily input: snapshots are Mondays; churn = no payment in the 60 days from the snapshot; only users who paid in the 60 days before are scored. Core input: month-end snapshots, no payment in the next 2 months, only users who have not already left.

**Features from the past only.** Every feature uses data before the snapshot. A test wipes all activity after a cut-off and checks that earlier features do not change, for both inputs.

**Daily features.** Days since last payment and last app open (capped at 90); payment count, card spend, p2p out and p2p in over 30 days; active payment days over 30 and 90 days; opens in the last 7 days and their trend against the daily average of the 21 days before; searches and chats over 30 days; chats per 100 opens; age and friends; android pay and photo events only if they happened before the snapshot.

**Time-based split with a gap.** The final test is the last 4 weeks of snapshots. Train only on snapshots whose label window ended before the test window starts.

**Backtest.** The same rule applied to consecutive 4-week test windows. Folds with fewer than 2,000 training rows are skipped. Every fold is written to `reports/backtest.csv` and printed, with the average per model also in `results.json`.

**Baseline first.** A simple rule (not active this month, few active months) sets the bar. Then logistic regression and gradient boosting.

**Metrics the team can use.** PR-AUC, recall and precision in the top 10% of scores (the list a team can call), Brier score for calibration. ROC-AUC from probabilities.

**Threshold from money, not 0.5.** Contact when p x offer success rate x customer value > offer cost. With placeholder values (cost 20, success 20%, value 300) the break-even probability is 33%. These inputs belong to the business.

**Explainability.** Permutation importance on the test month, and SHAP for per-user reasons if `shap` is installed.

## Run it

```sh
pip install -e ".[dev,shap]"
cp /path/to/df_*.csv data/       # core, transactions, activity_daily, users
python -m retention.run            # writes reports/ (figures, CSVs, results.json)
pytest -q                          # tests run on synthetic data, no real data needed
```

To walk through the pipeline step by step, open `notebook/retention_walkthrough.ipynb` (install with `pip install -e ".[notebook]"`). It calls the package functions and shows the data after each step. It is committed without outputs because they contain per-user rows: clear outputs before committing it.

Column names live in `src/retention/config.py`. If the export differs, the loader lists the columns it found.

## Limits

- Transactions before Nov 2017 cover only a handful of users, so early snapshots are small and the earliest backtest folds are skipped.
- The last usable snapshot is 60 days before the data ends (mid-June 2018), so recent behaviour is not in the test.
- `age`, `friends_on_monzo` and the event dates come from a users table exported at the end of the period. Friends and age are not point-in-time values, so they may carry a little future information.
- `offered_overdraft` is a 0/1 flag with no date in the users table, so it is not used in the daily input.
- Days since last app open is capped at 90 days (a 90-day activity lookback).
- p2p direction comes from the sign of the amount. The £1 test-transfer threshold is a judgment call, not a documented rule.
- Offer cost, success rate and customer value are placeholders.

## Layout

```
src/retention/
  config.py     settings and column names
  data.py       load the core table, active flag, observed window
  daily.py      weekly snapshots from the daily tables (DuckDB)
  cohorts.py    retention matrix, engagement by tenure, chats vs churn
  snapshots.py  month-end snapshots from the core table (fallback)
  model.py      time split, baseline, models, metrics, threshold
  run.py        end-to-end pipeline
notebook/       step-by-step walkthrough of the pipeline
tests/          synthetic data and tests (labels, leakage, split gap)
```

## Results

Filled in after running on the real data: see `reports/results.json` and the figures in `reports/`.
