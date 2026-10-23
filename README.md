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

**Churn label.** At each month-end snapshot: churn = no payment in the next 2 months. Only users who have not already left are scored.

**Features from the past only.** Every feature uses months up to the snapshot. A test wipes all activity after a cut-off and checks that earlier features do not change.

**Time-based split with a gap.** Test on the last snapshot month. Train only on snapshots whose 2-month label window had closed by then.

**Baseline first.** A simple rule (not active this month, few active months) sets the bar. Then logistic regression and gradient boosting.

**Metrics the team can use.** PR-AUC, recall and precision in the top 10% of scores (the list a team can call), Brier score for calibration. ROC-AUC from probabilities.

**Threshold from money, not 0.5.** Contact when p x offer success rate x customer value > offer cost. With placeholder values (cost 20, success 20%, value 300) the break-even probability is 33%. These inputs belong to the business.

**Explainability.** Permutation importance on the test month, and SHAP for per-user reasons if `shap` is installed.

## Run it

```sh
pip install -e ".[dev,shap]"
cp /path/to/df_core.csv data/
python -m retention.run            # writes reports/ (figures, CSVs, results.json)
pytest -q                          # tests run on synthetic data, no real data needed
```

Column names live in `src/retention/config.py`. If the export differs, the loader lists the columns it found.

## Layout

```
src/retention/
  config.py     settings and column names
  data.py       load, active flag, observed window
  cohorts.py    retention matrix, engagement by tenure, chats vs churn
  snapshots.py  leakage-safe features and labels
  model.py      time split, baseline, models, metrics, threshold
  run.py        end-to-end pipeline
tests/          synthetic data and tests (labels, leakage, split gap)
```

## Results

Filled in after running on the real data: see `reports/results.json` and the figures in `reports/`.
