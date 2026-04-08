"""
Deterministic dataset generator for data_clean_env.

Each task returns (dirty_df, clean_df).
- clean_df  : ground truth — all issues resolved
- dirty_df  : deep copy of clean with specific, known issues injected

All generation is seeded so the same seed → same dataset every time.
"""
from __future__ import annotations

import copy
import random
from datetime import date, timedelta
from typing import Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Robinson", "Clark",
]
DEPARTMENTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
STATUS_CHOICES = ["shipped", "cancelled", "delivered", "pending", "processing"]
CATEGORIES = ["Food", "Transport", "Entertainment", "Utilities", "Healthcare", "Gas Station"]
ACCOUNT_IDS = [f"ACC{str(i).zfill(4)}" for i in range(1, 21)]  # ACC0001..ACC0020


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _np_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_date(rng: random.Random, start: date, days: int) -> date:
    return start + timedelta(days=rng.randint(0, days))


def _phone_e164(rng: random.Random) -> str:
    """Generate a canonical E.164-style phone: +1XXXXXXXXXX"""
    area = rng.randint(200, 999)
    mid = rng.randint(100, 999)
    end = rng.randint(1000, 9999)
    return f"+1{area}{mid}{end}"


def _dirty_phone(rng: random.Random, canonical: str) -> str:
    """Return the same number in a messy format."""
    digits = canonical[2:]  # strip +1
    fmt = rng.choice(["parens", "dashes", "plain"])
    if fmt == "parens":
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if fmt == "dashes":
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits  # plain 10-digit


# ---------------------------------------------------------------------------
# Task 1 — Easy: "Clean Employee Records" (50 rows)
# ---------------------------------------------------------------------------

def _generate_easy(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng(seed)
    nprng = _np_rng(seed)

    n = 50
    start = date(2020, 1, 1)

    records = []
    for i in range(n):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        dept = rng.choice(DEPARTMENTS)
        salary = round(rng.uniform(45_000, 120_000), 2)
        hire_date = _random_date(rng, start, 3 * 365)
        records.append({
            "employee_id": i + 1,
            "name": f"{fname} {lname}",
            "department": dept,
            "salary": salary,
            "hire_date": hire_date.isoformat(),
        })

    clean_df = pd.DataFrame(records)

    # ---- Inject issues into dirty copy ----
    dirty = copy.deepcopy(records)

    # Issue 1: salary stored as string with "$" prefix — only 8 rows (not all)
    dollar_indices = rng.sample(range(n), 8)
    for idx in dollar_indices:
        dirty[idx]["salary"] = f"${dirty[idx]['salary']:.2f}"

    # Issue 2: Inconsistent casing in department — 8 rows
    case_indices = rng.sample(range(n), 8)
    for idx in case_indices:
        choice = rng.choice(["lower", "upper"])
        if choice == "lower":
            dirty[idx]["department"] = dirty[idx]["department"].lower()
        else:
            dirty[idx]["department"] = dirty[idx]["department"].upper()

    # Issue 3: Missing salary values (3 nulls)
    null_salary_indices = rng.sample(range(n), 3)
    for idx in null_salary_indices:
        dirty[idx]["salary"] = None

    # Issue 4: Missing department values (2 nulls)
    null_dept_indices = rng.sample([i for i in range(n) if i not in null_salary_indices], 2)
    for idx in null_dept_indices:
        dirty[idx]["department"] = None

    # Issue 5: 2 exact duplicate rows (not 3 — keeps structure score higher)
    dup_indices = rng.sample(range(n), 2)
    for idx in dup_indices:
        dirty.append(copy.deepcopy(dirty[idx]))

    dirty_df = pd.DataFrame(dirty)

    # Clean df: salary as float, department titlecase, no nulls, no dups
    # For the clean df we use original records (no injected issues)
    # Fill the missing salary with median, missing dept with mode
    clean_df["salary"] = clean_df["salary"].astype(float)
    clean_df["department"] = clean_df["department"].str.strip().str.title()

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Task 2 — Medium: "Fix Customer Orders Dataset" (100 rows)
# ---------------------------------------------------------------------------

def _generate_medium(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng(seed)
    n = 100
    start = date(2023, 1, 1)

    records = []
    for i in range(n):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        email = f"{fname.lower()}.{lname.lower()}{rng.randint(1,99)}@example.com"
        phone = _phone_e164(rng)
        order_date = _random_date(rng, start, 365)
        quantity = rng.randint(1, 20)
        unit_price = round(rng.uniform(5.0, 200.0), 2)
        order_total = round(quantity * unit_price, 2)
        status = rng.choice(STATUS_CHOICES)
        records.append({
            "order_id": 1000 + i,
            "customer_name": f"{fname} {lname}",
            "email": email,
            "phone": phone,
            "order_date": order_date.isoformat(),  # ISO format
            "quantity": quantity,
            "order_total": order_total,
            "status": status,
        })

    clean_df = pd.DataFrame(records)
    dirty = [dict(r) for r in records]

    # Issue 1: Mixed date formats — 50 rows (was 30)
    date_fmt_indices = rng.sample(range(n), 50)
    for idx in date_fmt_indices:
        d = date.fromisoformat(dirty[idx]["order_date"])
        fmt = rng.choice(["us", "short"])
        if fmt == "us":
            dirty[idx]["order_date"] = d.strftime("%m/%d/%Y")
        else:
            # "DD-Mon-YY"
            dirty[idx]["order_date"] = d.strftime("%d-%b-%y")

    # Issue 2: Inconsistent phone formats — all rows
    for r in dirty:
        r["phone"] = _dirty_phone(rng, r["phone"])

    # Issue 3: Missing email (8 nulls) — also remove those rows from clean
    null_email_idx = rng.sample(range(n), 8)
    for idx in null_email_idx:
        dirty[idx]["email"] = None

    # Issue 4: Negative order_total (8 rows, was 4)
    neg_idx = rng.sample(range(n), 8)
    for idx in neg_idx:
        dirty[idx]["order_total"] = -abs(dirty[idx]["order_total"])

    # Issue 5: Status typos — 20 rows (was 10)
    typos = {"shipped": "shiped", "cancelled": "cancled", "delivered": "deliverd"}
    typo_idx = rng.sample(range(n), 20)
    for idx in typo_idx:
        original = dirty[idx]["status"]
        if original in typos:
            dirty[idx]["status"] = typos[original]

    # Issue 6: String quantities — 10 rows (was 5)
    num_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    word_idx = rng.sample(range(n), 10)
    for idx in word_idx:
        qty = dirty[idx]["quantity"]
        if qty in num_words:
            dirty[idx]["quantity"] = num_words[qty]
        else:
            dirty[idx]["quantity"] = "many"  # fallback text

    # Issue 7: Outlier order_total (4 rows)
    outlier_idx = rng.sample(range(n), 4)
    for idx in outlier_idx:
        dirty[idx]["order_total"] = 999_999.99

    # Issue 8: 8 exact duplicate rows — removable by deduplicate operation
    dup_indices = rng.sample(range(n), 8)
    for idx in dup_indices:
        dirty.append(copy.deepcopy(dirty[idx]))

    dirty_df = pd.DataFrame(dirty)

    # Clean df: drop the null-email rows (same indices as injected into dirty),
    # clip order_total, cast quantity to int
    clean_df = clean_df.drop(index=null_email_idx).reset_index(drop=True)
    clean_df["order_total"] = clean_df["order_total"].clip(lower=0, upper=4000)
    clean_df["quantity"] = clean_df["quantity"].astype(int)

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Task 3 — Hard: "Reconcile Financial Transactions" (150 rows)
# ---------------------------------------------------------------------------

MERCHANTS = [
    "Whole Foods Market", "Shell Gas Station", "Netflix", "City Water Dept",
    "Starbucks Coffee", "Delta Airlines", "CVS Pharmacy", "Home Depot",
    "Uber Ride", "Amazon Prime", "McDonald's", "Spotify", "PG&E Utility",
    "Chevron Gas", "Planet Fitness",
]
TX_CATEGORIES = {
    "Whole Foods Market": "Food",
    "Shell Gas Station": "Gas Station",
    "Netflix": "Entertainment",
    "City Water Dept": "Utilities",
    "Starbucks Coffee": "Food",
    "Delta Airlines": "Transport",
    "CVS Pharmacy": "Healthcare",
    "Home Depot": "Utilities",
    "Uber Ride": "Transport",
    "Amazon Prime": "Entertainment",
    "McDonald's": "Food",
    "Spotify": "Entertainment",
    "PG&E Utility": "Utilities",
    "Chevron Gas": "Gas Station",
    "Planet Fitness": "Healthcare",
}


def _generate_hard(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng(seed)
    n = 150
    start = date(2024, 1, 1)

    records = []
    for i in range(n):
        account_id = rng.choice(ACCOUNT_IDS)
        merchant = rng.choice(MERCHANTS)
        category = TX_CATEGORIES[merchant]
        tx_date = _random_date(rng, start, 365)
        settle_date = tx_date + timedelta(days=rng.randint(0, 3))
        gross = round(rng.uniform(5.0, 500.0), 2)
        tax = round(gross * rng.uniform(0.05, 0.12), 2)
        net = round(gross - tax, 2)
        # debit/credit split: net is always debit; credit is 0 for purchases
        debit = net
        credit = 0.0
        total = round(debit + credit, 2)

        records.append({
            "tx_id": f"TX{str(i+1).zfill(5)}",
            "account_id": account_id,
            "tx_date": tx_date.isoformat(),
            "settlement_date": settle_date.isoformat(),
            "merchant": merchant,
            "category": category,
            "gross_amount": gross,
            "tax": tax,
            "net_amount": net,
            "debit": debit,
            "credit": credit,
            "total": total,
            "currency_raw": f"${gross:,.2f}",
        })

    clean_df = pd.DataFrame(records)
    dirty = [dict(r) for r in records]

    # Issue 1: Corrupt total for ALL rows (debit+credit != total everywhere)
    for r in dirty:
        r["total"] = round(r["total"] + rng.uniform(5, 50), 2)

    # Issue 2: Reverse settlement_date for ALL rows (settlement < tx_date)
    for r in dirty:
        tx = date.fromisoformat(r["tx_date"])
        days_back = rng.randint(1, 5)
        r["settlement_date"] = (tx - timedelta(days=days_back)).isoformat()

    # Issue 3: Currency in wrong format for ALL rows — use formats that never
    # match the clean "$X.XX" (no dollar_comma since amounts < 1000 look identical)
    for r in dirty:
        fmt = rng.choice(["usd_prefix", "plain"])
        if fmt == "usd_prefix":
            r["currency_raw"] = f"USD {r['gross_amount']:.2f}"
        else:
            r["currency_raw"] = str(r["gross_amount"])

    # Issue 4: Wrong category for ALL rows (invalid value)
    for r in dirty:
        r["category"] = "Unknown"

    # Issue 5: Missing net_amount (80 rows)
    missing_net_idx = rng.sample(range(n), 80)
    for idx in missing_net_idx:
        dirty[idx]["net_amount"] = None

    # Issue 6: Missing tax (60 rows)
    missing_tax_idx = rng.sample(range(n), 60)
    for idx in missing_tax_idx:
        dirty[idx]["tax"] = None

    # Issue 7: Missing debit (40 rows)
    missing_debit_idx = rng.sample(range(n), 40)
    for idx in missing_debit_idx:
        dirty[idx]["debit"] = None

    # Issue 8: Orphan account_ids (30 rows)
    orphan_idx = rng.sample(range(n), 30)
    for idx in orphan_idx:
        dirty[idx]["account_id"] = "ACC9999"

    # Issue 9: Real duplicate rows — 20 identical rows (same tx_id + all fields)
    # These are detected by df.duplicated() and removed by deduplicate operation
    dup_indices = rng.sample(range(n), 20)
    for idx in dup_indices:
        dirty.append(copy.deepcopy(dirty[idx]))

    dirty_df = pd.DataFrame(dirty)

    # Clean df: all values correct from original records; just normalize formats
    clean_df["total"] = (clean_df["debit"] + clean_df["credit"]).round(2)
    clean_df["currency_raw"] = clean_df["gross_amount"].apply(lambda x: f"${x:.2f}")
    clean_df["gross_amount"] = clean_df["gross_amount"].round(2)

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "easy": _generate_easy,
    "medium": _generate_medium,
    "hard": _generate_hard,
}

TASK_DESCRIPTIONS = {
    "easy": (
        "Clean a 50-row employee records dataset. Issues include: salary stored as "
        "strings with '$' prefix, inconsistent department casing (Engineering vs "
        "engineering vs ENGINEERING), missing salary values (fill with median), "
        "missing department values (fill with mode), and exact duplicate rows."
    ),
    "medium": (
        "Fix a 100-row customer orders dataset. Issues include: mixed date formats "
        "(ISO, US MM/DD/YYYY, DD-Mon-YY), inconsistent phone number formats, missing "
        "email addresses, negative order totals (data entry errors), status field typos "
        "('shiped', 'cancled', 'deliverd'), string quantities ('two', 'three'), and "
        "extreme outliers in order_total."
    ),
    "hard": (
        "Reconcile a 150-row financial transactions dataset. Issues include: "
        "ALL rows have debit+credit ≠ total (cross-column inconsistency), ALL rows have "
        "settlement_date before tx_date (temporal logic errors), ALL rows have wrong "
        "currency format ('USD X.XX' or plain float instead of '$X.XX'), ALL rows have "
        "category set to 'Unknown' (invalid), missing net_amount (80 rows), missing tax "
        "(60 rows), missing debit (40 rows), orphan account_ids (30 rows), and 20 exact "
        "duplicate rows."
    ),
}


def generate_task_data(task_id: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (dirty_df, clean_df) pair for the given task.

    Parameters
    ----------
    task_id : str
        One of 'easy', 'medium', 'hard'
    seed : int
        Random seed for reproducibility

    Returns
    -------
    (dirty_df, clean_df) : tuple of DataFrames
    """
    if task_id not in TASK_GENERATORS:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_GENERATORS)}")
    return TASK_GENERATORS[task_id](seed)
