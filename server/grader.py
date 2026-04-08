"""
Deterministic grading logic for data_clean_env.

grade_submission(current_df, clean_df, task_id) → dict with:
  score, completeness, accuracy, consistency, structure, details
All sub-scores are in [0.0, 1.0].

Design principles:
- Cell-level comparison for accuracy
- Sub-score components for rich training signal
- Always deterministic: same input → same scores
- Handles edge cases: wrong shape, extra/missing columns, empty df
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PHONE_E164_RE = re.compile(r"^\+1\d{10}$")
_CURRENCY_NUM_RE = re.compile(r"^\d+(\.\d{1,2})?$")
# Clean currency format: "$" followed by digits, optional ".XX" — no commas, no "USD" prefix
_CLEAN_CURR_RE = re.compile(r"^\$\d+(\.\d{2})?$")


def _safe_numeric(val: Any) -> float | None:
    """Try converting val to float; return None on failure."""
    try:
        return float(str(val).replace(",", "").replace("$", "").replace("USD ", "").strip())
    except (ValueError, TypeError):
        return None


def _normalize_str(val: Any) -> str:
    """Lowercase, strip whitespace."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _cells_equal(agent_val: Any, truth_val: Any, dtype_str: str) -> bool:
    """
    Compare one cell. For numeric types use tolerance; for strings use
    case-insensitive exact match after strip.
    """
    if pd.isna(agent_val) and pd.isna(truth_val):
        return True
    if pd.isna(agent_val) or pd.isna(truth_val):
        return False

    if "int" in dtype_str or "float" in dtype_str:
        a = _safe_numeric(agent_val)
        b = _safe_numeric(truth_val)
        if a is None or b is None:
            return False
        if b == 0:
            return abs(a - b) < 1e-4
        return abs(a - b) / (abs(b) + 1e-10) < 0.01  # 1 % relative tolerance
    else:
        return _normalize_str(agent_val) == _normalize_str(truth_val)


# ---------------------------------------------------------------------------
# Sub-score: Completeness
# ---------------------------------------------------------------------------

def _completeness_score(agent_df: pd.DataFrame, clean_df: pd.DataFrame, original_null_counts: Dict[str, int]) -> float:
    """
    % of originally-null cells that are now non-null in agent_df.
    """
    total_original_nulls = sum(original_null_counts.values())
    if total_original_nulls == 0:
        return 1.0

    # Count remaining nulls in agent_df for the same columns
    remaining = 0
    for col, orig_count in original_null_counts.items():
        if orig_count == 0:
            continue
        if col not in agent_df.columns:
            remaining += orig_count
        else:
            remaining += int(agent_df[col].isna().sum())

    fixed = total_original_nulls - remaining
    return max(0.0, min(1.0, fixed / total_original_nulls))


# ---------------------------------------------------------------------------
# Sub-score: Accuracy
# ---------------------------------------------------------------------------

def _accuracy_score(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Cell-by-cell comparison between agent_df and clean_df.

    Uses key-based merge (inner join on primary key) when the first shared
    column is unique in clean_df — this handles duplicate rows and row-count
    differences correctly.  Falls back to positional alignment otherwise.
    """
    shared_cols = [c for c in clean_df.columns if c in agent_df.columns]
    if not shared_cols:
        return 0.0

    n_clean = len(clean_df)
    if n_clean == 0:
        return 0.0

    key_col = shared_cols[0]
    value_cols = shared_cols[1:]

    # Try key-based merge if first column is a unique key in clean_df
    use_key = bool(value_cols)
    if use_key:
        try:
            use_key = int(clean_df[key_col].nunique()) == n_clean
        except Exception:
            use_key = False

    if use_key:
        try:
            # Deduplicate agent by key so each clean row has at most one match
            agent_deduped = agent_df[shared_cols].drop_duplicates(subset=[key_col], keep="first")
            merged = pd.merge(
                clean_df[shared_cols],
                agent_deduped,
                on=key_col,
                how="left",
                suffixes=("_clean", "_agent"),
            )
            total_cells = len(merged) * len(value_cols)
            if total_cells == 0:
                return 1.0
            matching = 0
            for col in value_cols:
                dtype_str = str(clean_df[col].dtype)
                c_col = f"{col}_clean"
                a_col = f"{col}_agent"
                for c_val, a_val in zip(merged[c_col], merged[a_col]):
                    if _cells_equal(a_val, c_val, dtype_str):
                        matching += 1
            return matching / total_cells
        except Exception:
            pass  # fall through to positional

    # Positional fallback (sorts by key column, truncates to min row count)
    n_rows = min(len(agent_df), n_clean)
    if n_rows == 0:
        return 0.0
    try:
        agent_s = agent_df[shared_cols].sort_values(key_col).reset_index(drop=True).iloc[:n_rows]
        clean_s = clean_df[shared_cols].sort_values(key_col).reset_index(drop=True).iloc[:n_rows]
    except Exception:
        agent_s = agent_df[shared_cols].reset_index(drop=True).iloc[:n_rows]
        clean_s = clean_df[shared_cols].reset_index(drop=True).iloc[:n_rows]

    total_cells = n_rows * len(shared_cols)
    matching = sum(
        _cells_equal(a, t, str(clean_df[col].dtype))
        for col in shared_cols
        for a, t in zip(agent_s[col], clean_s[col])
    )
    return matching / total_cells if total_cells > 0 else 0.0


# ---------------------------------------------------------------------------
# Sub-score: Consistency (format checks per task)
# ---------------------------------------------------------------------------

def _consistency_score_easy(agent_df: pd.DataFrame) -> float:
    checks = []

    # 1. Department casing should be titlecase
    if "department" in agent_df.columns:
        dept_vals = agent_df["department"].dropna().astype(str)
        if len(dept_vals) > 0:
            pct = sum(v == v.title() for v in dept_vals) / len(dept_vals)
            checks.append(pct)

    # 2. salary should be numeric (float)
    if "salary" in agent_df.columns:
        salary_vals = agent_df["salary"].dropna()
        if len(salary_vals) > 0:
            numeric_count = sum(
                True for v in salary_vals if _safe_numeric(v) is not None
            )
            checks.append(numeric_count / len(salary_vals))

    # 3. hire_date should be ISO format
    if "hire_date" in agent_df.columns:
        dates = agent_df["hire_date"].dropna().astype(str)
        if len(dates) > 0:
            pct = sum(bool(_ISO_DATE_RE.match(d)) for d in dates) / len(dates)
            checks.append(pct)

    return sum(checks) / len(checks) if checks else 1.0


def _consistency_score_medium(agent_df: pd.DataFrame) -> float:
    checks = []

    # 1. order_date: ISO format
    if "order_date" in agent_df.columns:
        dates = agent_df["order_date"].dropna().astype(str)
        if len(dates) > 0:
            pct = sum(bool(_ISO_DATE_RE.match(d)) for d in dates) / len(dates)
            checks.append(pct)

    # 2. phone: E.164 format
    if "phone" in agent_df.columns:
        phones = agent_df["phone"].dropna().astype(str)
        if len(phones) > 0:
            pct = sum(bool(_PHONE_E164_RE.match(p)) for p in phones) / len(phones)
            checks.append(pct)

    # 3. order_total: numeric and non-negative
    if "order_total" in agent_df.columns:
        totals = agent_df["order_total"].dropna()
        if len(totals) > 0:
            pct = sum(
                _safe_numeric(v) is not None and _safe_numeric(v) >= 0
                for v in totals
            ) / len(totals)
            checks.append(pct)

    # 4. quantity: integer-like values
    if "quantity" in agent_df.columns:
        qtys = agent_df["quantity"].dropna()
        if len(qtys) > 0:
            pct = sum(
                str(v).lstrip("-").isdigit() for v in qtys
            ) / len(qtys)
            checks.append(pct)

    # 5. status: valid values
    valid_status = {"shipped", "cancelled", "delivered", "pending", "processing"}
    if "status" in agent_df.columns:
        statuses = agent_df["status"].dropna().str.lower()
        if len(statuses) > 0:
            pct = sum(s in valid_status for s in statuses) / len(statuses)
            checks.append(pct)

    return sum(checks) / len(checks) if checks else 1.0


def _consistency_score_hard(agent_df: pd.DataFrame) -> float:
    checks = []

    # 1. total = debit + credit (within tolerance)
    if all(c in agent_df.columns for c in ["total", "debit", "credit"]):
        rows_checked = 0
        rows_consistent = 0
        for _, row in agent_df.iterrows():
            d = _safe_numeric(row.get("debit"))
            c = _safe_numeric(row.get("credit"))
            t = _safe_numeric(row.get("total"))
            if d is not None and c is not None and t is not None:
                rows_checked += 1
                if abs((d + c) - t) < 0.05:
                    rows_consistent += 1
        if rows_checked > 0:
            checks.append(rows_consistent / rows_checked)

    # 2. settlement_date >= tx_date
    if all(c in agent_df.columns for c in ["tx_date", "settlement_date"]):
        rows_checked = 0
        rows_ok = 0
        for _, row in agent_df.iterrows():
            td = str(row.get("tx_date", ""))
            sd = str(row.get("settlement_date", ""))
            if _ISO_DATE_RE.match(td) and _ISO_DATE_RE.match(sd):
                rows_checked += 1
                if sd >= td:
                    rows_ok += 1
        if rows_checked > 0:
            checks.append(rows_ok / rows_checked)

    # 3. currency_raw: must be in clean "$X.XX" format (no commas, no "USD" prefix)
    if "currency_raw" in agent_df.columns:
        vals = agent_df["currency_raw"].dropna().astype(str)
        if len(vals) > 0:
            pct = sum(bool(_CLEAN_CURR_RE.match(v)) for v in vals) / len(vals)
            checks.append(pct)

    # 4. net_amount present and not null
    if "net_amount" in agent_df.columns:
        present = agent_df["net_amount"].notna().sum() / max(len(agent_df), 1)
        checks.append(present)

    # 5. valid category
    valid_cats = {"food", "transport", "entertainment", "utilities", "healthcare", "gas station"}
    if "category" in agent_df.columns:
        cats = agent_df["category"].dropna().str.lower()
        if len(cats) > 0:
            pct = sum(c in valid_cats for c in cats) / len(cats)
            checks.append(pct)

    return sum(checks) / len(checks) if checks else 1.0


_CONSISTENCY_FNS = {
    "easy": _consistency_score_easy,
    "medium": _consistency_score_medium,
    "hard": _consistency_score_hard,
}


# ---------------------------------------------------------------------------
# Sub-score: Structure
# ---------------------------------------------------------------------------

def _structure_score(agent_df: pd.DataFrame, clean_df: pd.DataFrame, original_dup_count: int) -> float:
    checks: List[float] = []

    # 1. Row count: agent should have roughly the same rows as clean
    # Penalise over-deletion (too few rows) or no dedup (too many rows)
    expected_rows = len(clean_df)
    actual_rows = len(agent_df)
    if expected_rows == 0:
        checks.append(1.0 if actual_rows == 0 else 0.0)
    else:
        ratio = actual_rows / expected_rows
        # Score 1.0 if within 5%, decreasing linearly to 0 at 50% off
        checks.append(max(0.0, 1.0 - abs(ratio - 1.0) * 2.0))

    # 2. Duplicate count: should be 0
    dup_count = int(agent_df.duplicated().sum())
    if original_dup_count > 0:
        checks.append(1.0 if dup_count == 0 else max(0.0, 1.0 - dup_count / original_dup_count))
    else:
        checks.append(1.0)

    # 3. Column presence: all clean columns should exist
    missing_cols = set(clean_df.columns) - set(agent_df.columns)
    col_score = 1.0 - len(missing_cols) / len(clean_df.columns)
    checks.append(max(0.0, col_score))

    # 4. Dtype compatibility for numeric columns
    numeric_cols = [c for c in clean_df.columns if "int" in str(clean_df[c].dtype) or "float" in str(clean_df[c].dtype)]
    if numeric_cols:
        ok = 0
        for col in numeric_cols:
            if col not in agent_df.columns:
                continue
            sample = agent_df[col].dropna().head(5)
            if all(_safe_numeric(v) is not None for v in sample):
                ok += 1
        checks.append(ok / len(numeric_cols))

    return sum(checks) / len(checks) if checks else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_submission(
    current_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    task_id: str,
    original_null_counts: Dict[str, int] | None = None,
    original_dup_count: int = 0,
) -> Dict:
    """
    Grade agent's current_df against the ground truth clean_df.

    Returns
    -------
    {
        'score': float,           # 0.0-1.0 overall (weighted average)
        'completeness': float,
        'accuracy': float,
        'consistency': float,
        'structure': float,
        'details': str,           # human-readable breakdown
    }
    """
    if original_null_counts is None:
        original_null_counts = {}

    # Guard: empty agent df
    if current_df is None or len(current_df) == 0:
        return {
            "score": 0.001,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "structure": 0.0,
            "details": "Agent dataframe is empty.",
        }

    completeness = _completeness_score(current_df, clean_df, original_null_counts)
    accuracy = _accuracy_score(current_df, clean_df)
    consistency_fn = _CONSISTENCY_FNS.get(task_id, _consistency_score_easy)
    consistency = consistency_fn(current_df)
    structure = _structure_score(current_df, clean_df, original_dup_count)

    # Clamp all to (0.001, 0.999) — strictly exclusive of 0 and 1
    completeness = max(0.001, min(0.999, completeness))
    accuracy = max(0.001, min(0.999, accuracy))
    consistency = max(0.001, min(0.999, consistency))
    structure = max(0.001, min(0.999, structure))

    score = (0.25 * completeness + 0.25 * accuracy + 0.25 * consistency + 0.25 * structure)
    score = max(0.001, min(0.999, score))

    details = (
        f"Completeness: {completeness:.3f} | "
        f"Accuracy: {accuracy:.3f} | "
        f"Consistency: {consistency:.3f} | "
        f"Structure: {structure:.3f} | "
        f"Overall: {score:.3f}"
    )

    return {
        "score": round(score, 6),
        "completeness": round(completeness, 6),
        "accuracy": round(accuracy, 6),
        "consistency": round(consistency, 6),
        "structure": round(structure, 6),
        "details": details,
    }


def compute_quality_issues(df: pd.DataFrame, task_id: str) -> List[str]:
    """
    Return a list of human-readable issue descriptions found in df.
    Used to populate the 'issues_found' field in the observation.
    """
    issues: List[str] = []

    # Universal checks
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            issues.append(f"Column '{col}' has {cnt} missing value(s)")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate row(s) found")

    if task_id == "easy":
        if "salary" in df.columns:
            # Check for dollar-sign strings
            sample = df["salary"].dropna().astype(str)
            dollar_count = sum(1 for v in sample if str(v).startswith("$"))
            if dollar_count > 0:
                issues.append(f"'salary' column has {dollar_count} values with '$' prefix (not numeric)")

        if "department" in df.columns:
            dept_vals = df["department"].dropna().astype(str)
            non_title = sum(1 for v in dept_vals if v != v.title())
            if non_title > 0:
                issues.append(f"'department' column has {non_title} inconsistently cased values")

    elif task_id == "medium":
        if "order_date" in df.columns:
            dates = df["order_date"].dropna().astype(str)
            non_iso = sum(1 for d in dates if not _ISO_DATE_RE.match(d))
            if non_iso > 0:
                issues.append(f"'order_date' has {non_iso} non-ISO date format(s)")

        if "phone" in df.columns:
            phones = df["phone"].dropna().astype(str)
            non_e164 = sum(1 for p in phones if not _PHONE_E164_RE.match(p))
            if non_e164 > 0:
                issues.append(f"'phone' has {non_e164} non-E.164 format(s)")

        if "order_total" in df.columns:
            negatives = sum(1 for v in df["order_total"].dropna() if _safe_numeric(v) is not None and _safe_numeric(v) < 0)
            if negatives > 0:
                issues.append(f"'order_total' has {negatives} negative value(s)")

        valid_status = {"shipped", "cancelled", "delivered", "pending", "processing"}
        if "status" in df.columns:
            bad_status = sum(
                1 for v in df["status"].dropna().str.lower()
                if v not in valid_status
            )
            if bad_status > 0:
                issues.append(f"'status' column has {bad_status} typo/invalid value(s)")

        if "quantity" in df.columns:
            non_int = sum(
                1 for v in df["quantity"].dropna()
                if not str(v).lstrip("-").isdigit()
            )
            if non_int > 0:
                issues.append(f"'quantity' has {non_int} non-integer value(s)")

        if "order_total" in df.columns:
            big = sum(1 for v in df["order_total"].dropna() if _safe_numeric(v) is not None and _safe_numeric(v) > 4000)
            if big > 0:
                issues.append(f"'order_total' has {big} extreme outlier(s) (>4000)")

    elif task_id == "hard":
        if all(c in df.columns for c in ["debit", "credit", "total"]):
            inconsistent = 0
            for _, row in df.iterrows():
                d = _safe_numeric(row.get("debit"))
                c = _safe_numeric(row.get("credit"))
                t = _safe_numeric(row.get("total"))
                if d is not None and c is not None and t is not None:
                    if abs((d + c) - t) >= 0.05:
                        inconsistent += 1
            if inconsistent > 0:
                issues.append(f"{inconsistent} rows where debit+credit != total")

        if all(c in df.columns for c in ["tx_date", "settlement_date"]):
            temporal_errors = 0
            for _, row in df.iterrows():
                td = str(row.get("tx_date", ""))
                sd = str(row.get("settlement_date", ""))
                if _ISO_DATE_RE.match(td) and _ISO_DATE_RE.match(sd):
                    if sd < td:
                        temporal_errors += 1
            if temporal_errors > 0:
                issues.append(f"{temporal_errors} rows where settlement_date < tx_date")

        if "currency_raw" in df.columns:
            bad_curr = sum(
                1 for v in df["currency_raw"].dropna().astype(str)
                if not _CLEAN_CURR_RE.match(v)
            )
            if bad_curr > 0:
                issues.append(f"'currency_raw' has {bad_curr} wrong format(s) (expected $X.XX)")

        if "category" in df.columns:
            valid_cats = {"food", "transport", "entertainment", "utilities", "healthcare", "gas station"}
            bad_cats = sum(1 for v in df["category"].dropna() if str(v).lower() not in valid_cats)
            if bad_cats > 0:
                issues.append(f"'category' has {bad_cats} invalid value(s)")

        for null_col in ["net_amount", "tax", "debit"]:
            if null_col in df.columns:
                cnt = df[null_col].isna().sum()
                if cnt > 0:
                    issues.append(f"'{null_col}' has {cnt} missing value(s)")

        valid_accounts = {f"ACC{str(i).zfill(4)}" for i in range(1, 21)}
        if "account_id" in df.columns:
            orphans = sum(1 for v in df["account_id"].dropna() if str(v) not in valid_accounts)
            if orphans > 0:
                issues.append(f"{orphans} orphan account_id(s) not in valid accounts list")

    return issues
