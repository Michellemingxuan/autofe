"""A synthetic use case shaped like CDSS, for building the agent without the real data.

It writes everything a run reads, in the layout a real use case has:

    <root>/train.csv valid.csv test.csv        model database, split by date
    <root>/screen_train.csv screen_valid.csv   the screen's rows
    <root>/few_shot.csv                        labelled example rows, with `batch`
    <root>/column_descriptions.json
    <root>/task_context.md
    <root>/additional_data/
        spends.parquet   + spends_data_sample.json
        payments.parquet + payments_data_sample.json
        balances.parquet + balances_data_sample.json
        wwcas_synthetic_flagged.csv            the CAS scope

Ids follow the real form ``<customer_id>_<dt>_<marker>``; ``dt`` is the row's
as-of date. Test holds the latest months, so it is out of time.

Two things are planted, so a run can be checked rather than eyeballed:

* **Signal outside the base set.** Default risk depends on how much of the last
  90 days' spend the customer paid back. No base column carries it; an L1
  feature built from payments and spends through a point-in-time linkage should
  verify.
* **A leak.** Defaulters get ``collections`` spends dated *after* their as-of
  date. A feature that ignores the point-in-time rule finds them and looks
  brilliant; one built through a correct linkage never sees them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["generate", "PLANTED_WINDOW_DAYS"]

PLANTED_WINDOW_DAYS = 90
_MONTHS = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
_HISTORY_START = pd.Timestamp("2023-07-01")
_HISTORY_END = pd.Timestamp("2025-03-01")

BASE_DESCRIPTIONS = {
    "credit_limit": "Total credit line on the account, USD",
    "utilization": "Revolving balance divided by credit limit at the as-of date",
    "tenure_months": "Months since the account was opened",
    "num_delinq_12m": "Count of delinquent cycles in the last 12 months",
    "bureau_score": "External bureau risk score; higher is safer",
    "income_est": "Estimated annual income, USD",
    "num_accounts": "Number of open tradelines at the bureau",
    "revolving_bal": "Revolving balance at the as-of date, USD",
}


def _sample_json(frame: pd.DataFrame, descriptions: dict[str, str], n: int = 5) -> dict:
    """The format the real sources use: {column: [description, [sample values]]}."""
    head = frame.head(n)
    out = {}
    for col, text in descriptions.items():
        values = head[col]
        if pd.api.types.is_datetime64_any_dtype(values):
            values = values.dt.strftime("%Y-%m-%d")
        out[col] = [text, [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                           for v in values]]
    return out


def generate(root: str | Path, n_customers: int = 6000, seed: int = 0) -> Path:
    """Write the synthetic use case under ``root`` and return it."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    extra = root / "additional_data"
    extra.mkdir(parents=True, exist_ok=True)

    customers = np.arange(100000, 100000 + n_customers)
    # How much of their spend each customer pays back; the hidden driver.
    payback = np.clip(rng.beta(4, 1.5, n_customers), 0.05, 1.0)
    spend_rate = rng.gamma(2.0, 0.25, n_customers)          # spends per day
    spend_size = rng.lognormal(3.5, 0.6, n_customers)       # USD per spend

    # --- histories ---------------------------------------------------------
    days = (_HISTORY_END - _HISTORY_START).days
    counts = rng.poisson(spend_rate * days)
    cust_idx = np.repeat(np.arange(n_customers), counts)
    spends = pd.DataFrame({
        "customer_id": customers[cust_idx].astype(str),
        "event_dt": _HISTORY_START + pd.to_timedelta(rng.integers(0, days, len(cust_idx)), "D"),
        "amount": np.round(rng.lognormal(np.log(spend_size[cust_idx]), 0.5), 2),
        "mcc_group": rng.choice(["grocery", "travel", "fuel", "retail", "services"],
                                len(cust_idx), p=[0.3, 0.1, 0.2, 0.3, 0.1]),
    })

    # Monthly statements: pay back `payback` of the month's spend, with noise.
    month = spends["event_dt"].dt.to_period("M")
    monthly = spends.groupby(["customer_id", month])["amount"].sum().reset_index()
    pay_share = pd.Series(payback, index=customers.astype(str))
    monthly["amount"] = np.round(
        monthly["amount"] * pay_share.loc[monthly["customer_id"]].to_numpy()
        * rng.lognormal(0, 0.15, len(monthly)), 2)
    payments = pd.DataFrame({
        "customer_id": monthly["customer_id"],
        "event_dt": monthly["event_dt"].dt.to_timestamp(how="end").dt.normalize()
                    + pd.to_timedelta(rng.integers(5, 20, len(monthly)), "D"),
        "amount": monthly["amount"],
        "returned": (rng.random(len(monthly)) < 0.01).astype(int),
    })
    balances = (spends.groupby(["customer_id", month])["amount"].sum()
                .groupby(level=0).cumsum().reset_index())
    balances["balance"] = np.round(balances.pop("amount") * 0.05, 2)
    balances["event_dt"] = balances.pop("event_dt").dt.to_timestamp()

    # --- model database ----------------------------------------------------
    n_snap = 2
    rows = pd.DataFrame({
        "customer_id": np.repeat(customers, n_snap).astype(str),
        "dt": np.concatenate([rng.choice(_MONTHS, n_snap, replace=False)
                              for _ in customers]),
    })
    cust = np.repeat(np.arange(n_customers), n_snap)
    rows["id"] = (rows["customer_id"] + "_" + rows["dt"].dt.strftime("%Y%m%d") + "_"
                  + rng.choice(["A", "B"], len(rows)))

    # The planted quantity, computed point-in-time from the histories.
    window = pd.Timedelta(days=PLANTED_WINDOW_DAYS)

    def _window_sum(events: pd.DataFrame) -> pd.Series:
        joined = rows[["id", "customer_id", "dt"]].merge(events, on="customer_id")
        keep = (joined["event_dt"] < joined["dt"]) & (joined["event_dt"] >= joined["dt"] - window)
        return joined[keep].groupby("id")["amount"].sum().reindex(rows["id"]).fillna(0.0)

    paid = _window_sum(payments).to_numpy()
    spent = _window_sum(spends).to_numpy()
    ratio = np.clip(paid / np.maximum(spent, 1.0), 0, 2)

    risk = rng.normal(0, 1, n_customers)[cust]                 # what base sees
    rows["credit_limit"] = np.round(rng.lognormal(9, 0.5, len(rows)), -2)
    rows["utilization"] = np.clip(0.35 + 0.12 * risk + rng.normal(0, 0.15, len(rows)), 0, 1.2)
    rows["tenure_months"] = rng.integers(3, 240, len(rows))
    rows["num_delinq_12m"] = rng.poisson(np.exp(-1.5 + 0.6 * risk))
    rows["bureau_score"] = np.round(700 - 45 * risk + rng.normal(0, 30, len(rows)))
    rows["income_est"] = np.round(rng.lognormal(11, 0.4, len(rows)), -2)
    rows["num_accounts"] = rng.integers(1, 15, len(rows))
    rows["revolving_bal"] = np.round(rows["utilization"] * rows["credit_limit"], 2)

    logit = -3.2 + 0.9 * risk - 3.0 * (ratio - 0.7) + rng.normal(0, 0.3, len(rows))
    rows["default"] = (rng.random(len(rows)) < 1 / (1 + np.exp(-logit))).astype(int)

    # The leak: collections activity after the as-of date, for defaulters only.
    bad = rows[rows["default"] == 1]
    leak = pd.DataFrame({
        "customer_id": bad["customer_id"].to_numpy(),
        "event_dt": bad["dt"].to_numpy() + pd.to_timedelta(rng.integers(1, 60, len(bad)), "D"),
        "amount": np.round(rng.lognormal(5, 0.5, len(bad)), 2),
        "mcc_group": "collections",
    })
    spends = pd.concat([spends, leak], ignore_index=True).sort_values(
        ["customer_id", "event_dt"], ignore_index=True)

    # --- splits: by date, test last (out of time) --------------------------
    table = rows.drop(columns=["customer_id", "dt"])
    table = table[["id", "default", *BASE_DESCRIPTIONS]]
    split = np.where(rows["dt"] < "2024-09-01", "train",
                     np.where(rows["dt"] < "2024-11-01", "valid", "test"))
    for name in ("train", "valid", "test"):
        table[split == name].to_csv(root / f"{name}.csv", index=False)
    table[split == "train"].to_csv(root / "screen_train.csv", index=False)
    table[split == "valid"].to_csv(root / "screen_valid.csv", index=False)

    train = table[split == "train"]
    shots = pd.concat([
        train[train["default"] == 1].sample(4 * 3, random_state=seed),
        train[train["default"] == 0].sample(4 * 3, random_state=seed),
    ])
    shots["batch"] = np.tile(np.arange(4), 6)
    shots.sort_values("batch").to_csv(root / "few_shot.csv", index=False)

    (root / "column_descriptions.json").write_text(json.dumps(
        {"id": "Row key: <customer_id>_<dt>_<marker>; dt (YYYYMMDD) is the as-of date",
         "default": "1 if the customer defaults within 12 months of the as-of date",
         **BASE_DESCRIPTIONS}, indent=2))
    (root / "task_context.md").write_text(
        "Synthetic small-business card portfolio. Each row is a customer at an "
        "as-of date; the model predicts default within 12 months. The incumbent "
        "model uses bureau and account-level columns only; transaction histories "
        "(spends, payments, balances) are not yet used.\n")

    # --- additional data, in the real sample format --------------------------
    for name, frame, desc in (
        ("spends", spends, {
            "customer_id": "Customer ID; joins to the customer part of the model id",
            "event_dt": "Date of the spend (YYYY-MM-DD)",
            "amount": "Spend amount, USD",
            "mcc_group": "Merchant category group"}),
        ("payments", payments, {
            "customer_id": "Customer ID; joins to the customer part of the model id",
            "event_dt": "Date the payment was made (YYYY-MM-DD)",
            "amount": "Payment amount, USD",
            "returned": "1 if the payment was returned (e.g. NSF), else 0"}),
        ("balances", balances, {
            "customer_id": "Customer ID; joins to the customer part of the model id",
            "event_dt": "First day of the statement month",
            "balance": "Statement balance, USD"}),
    ):
        frame.to_parquet(extra / f"{name}.parquet", index=False)
        (extra / f"{name}_data_sample.json").write_text(
            json.dumps(_sample_json(frame, desc), indent=2, default=str))

    # Shaped like the real CAS exports: a row key, a customer identifier, and the
    # partition date a query must filter on.
    scope = [("cas_pkey", "wwcas_synthetic", "cas primary key", "int64", "no", "yes",
              "NOT IN CDSS_G9 VI/SHAP", None),
             ("customer_id", "wwcas_synthetic", "Customer ID - joins to the customer part of "
              "the model id", "string", "no", "no", "NOT IN CDSS_G9 VI/SHAP", None),
             ("trans_dt", "wwcas_synthetic", "Transaction date", "date", "yes", "no",
              "NOT IN CDSS_G9 VI/SHAP", None)]
    for col, text in BASE_DESCRIPTIONS.items():
        scope.append((col, "wwcas_synthetic", text, "float64", "no", "no",
                      "USED - CDSS_G9 VI/SHAP", col.upper()))
    for col, text in (("auth_decline_cnt_30d", "Authorization declines in the last 30 days"),
                      ("cash_adv_amt_90d", "Cash advance amount in the last 90 days"),
                      ("merchant_country", "Merchant country code")):
        scope.append((col, "wwcas_synthetic", text, "string", "no", "no",
                      "NOT IN CDSS_G9 VI/SHAP", None))
    pd.DataFrame(scope, columns=["NAME", "TABLE NAME", "DESCRIPTION", "TYPE", "PARTITION",
                                 "PRIMARY", "CDSS_G9_FLAG", "CDSS_G9_MATCHED_VARIABLE"]
                 ).to_csv(extra / "wwcas_synthetic_flagged.csv", index=False)
    return root


# The repo's data/ folder, wherever the generator is run from - a relative
# default once put a second copy under src/data when run from src/.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic_agent"


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(generate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
