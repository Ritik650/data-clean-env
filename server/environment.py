"""
Core DataCleanEnvironment class.

Implements the OpenEnv Environment interface:
  reset(**kwargs)  -> DataCleanObservation
  step(action)     -> DataCleanObservation
  state            -> DataCleanState   (property)
  reward           -> float            (property)
  done             -> bool             (property)

The framework's create_app() reads env.reward and env.done after each step()
call and wraps the observation into a StepResult.
"""
from __future__ import annotations

import copy
import re
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from server.data_generator import (
    TASK_DESCRIPTIONS,
    generate_task_data,
)
from server.grader import compute_quality_issues, grade_submission
from models import DataCleanAction, DataCleanObservation, DataCleanState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 15
PREVIEW_ROWS = 10

# Deterministic seed offsets for multi-seed variants
_SEED_VARIANTS = [42, 137, 256]


# ---------------------------------------------------------------------------
# Helper: human-readable data preview
# ---------------------------------------------------------------------------

def _df_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> str:
    return df.head(n).to_string(index=False)


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def _execute_action(df: pd.DataFrame, action: DataCleanAction) -> tuple[pd.DataFrame, str, Optional[str]]:
    """
    Apply the cleaning operation to df.

    Returns
    -------
    (new_df, result_message, error_message)
    error_message is None on success.
    """
    op = action.operation
    col = action.column
    params = action.params or {}

    # -- submit: no-op, just trigger final grading --
    if op == "submit":
        return df, "Submitted for final grading.", None

    # Column-required operations
    NEEDS_COLUMN = {
        "fill_missing", "replace_value", "cast_type",
        "normalize_format", "clip_outliers", "rename_column", "drop_column",
        "fix_inconsistency",
    }

    if op in NEEDS_COLUMN:
        if col is None:
            return df, "", "column is required for operation '{}'".format(op)
        if col not in df.columns:
            return df, "", "Column '{}' not found in dataframe".format(col)

    try:
        df = df.copy()  # always work on a copy

        # ---------------------------------------------------------------
        if op == "fill_missing":
            strategy = params.get("strategy", "value")
            value = params.get("value")
            if strategy == "mean":
                fill_val = pd.to_numeric(df[col], errors="coerce").mean()
            elif strategy == "median":
                fill_val = pd.to_numeric(df[col], errors="coerce").median()
            elif strategy == "mode":
                mode_series = df[col].mode()
                fill_val = mode_series.iloc[0] if len(mode_series) > 0 else value
            else:
                fill_val = value
            nulls_before = int(df[col].isna().sum())
            df[col] = df[col].fillna(fill_val)
            nulls_after = int(df[col].isna().sum())
            filled = nulls_before - nulls_after
            return df, f"Filled {filled} missing value(s) in '{col}' using strategy='{strategy}'.", None

        # ---------------------------------------------------------------
        elif op == "drop_rows":
            condition = params.get("condition", "nulls")
            rows_before = len(df)
            if condition == "duplicates":
                df = df.drop_duplicates().reset_index(drop=True)
            elif condition == "nulls":
                subset = [col] if col else None
                df = df.dropna(subset=subset).reset_index(drop=True)
            elif condition == "expression":
                expr = params.get("expression", "")
                try:
                    mask = df.eval(expr)
                    df = df[~mask].reset_index(drop=True)
                except Exception as e:
                    return df, "", f"Invalid expression '{expr}': {e}"
            else:
                return df, "", f"Unknown drop_rows condition: '{condition}'"
            rows_after = len(df)
            return df, f"Dropped {rows_before - rows_after} row(s) (condition='{condition}').", None

        # ---------------------------------------------------------------
        elif op == "deduplicate":
            rows_before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            rows_after = len(df)
            return df, f"Removed {rows_before - rows_after} duplicate row(s).", None

        # ---------------------------------------------------------------
        elif op == "replace_value":
            old_val = params.get("old")
            new_val = params.get("new")
            if old_val is None:
                return df, "", "params.old is required for replace_value"
            count = int((df[col].astype(str) == str(old_val)).sum())
            df[col] = df[col].astype(str).replace(str(old_val), str(new_val))
            return df, f"Replaced {count} occurrence(s) of '{old_val}' with '{new_val}' in '{col}'.", None

        # ---------------------------------------------------------------
        elif op == "cast_type":
            dtype = params.get("dtype", "str")
            rows_before = df[col].notna().sum()
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                # Strip currency symbols before casting
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str":
                df[col] = df[col].astype(str)
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce").dt.date.astype(str)
            return df, f"Cast column '{col}' to type '{dtype}'.", None

        # ---------------------------------------------------------------
        elif op == "normalize_format":
            fmt = params.get("format", "lowercase")
            changed = 0
            if fmt == "uppercase":
                original = df[col].copy()
                df[col] = df[col].astype(str).str.upper()
                changed = int((df[col] != original.astype(str)).sum())
            elif fmt == "lowercase":
                original = df[col].copy()
                df[col] = df[col].astype(str).str.lower()
                changed = int((df[col] != original.astype(str)).sum())
            elif fmt == "titlecase":
                original = df[col].copy()
                df[col] = df[col].astype(str).str.title()
                changed = int((df[col] != original.astype(str)).sum())
            elif fmt == "date_iso":
                def _to_iso(v: Any) -> str:
                    if pd.isna(v):
                        return v
                    try:
                        return pd.to_datetime(str(v), infer_datetime_format=True).date().isoformat()
                    except Exception:
                        return str(v)
                original = df[col].copy()
                df[col] = df[col].apply(_to_iso)
                changed = int((df[col] != original.astype(str)).sum())
            elif fmt == "phone_e164":
                def _to_e164(v: Any) -> str:
                    if pd.isna(v) or str(v).strip() == "":
                        return v
                    digits = re.sub(r"\D", "", str(v))
                    if digits.startswith("1") and len(digits) == 11:
                        digits = digits[1:]
                    if len(digits) == 10:
                        return f"+1{digits}"
                    return str(v)
                original = df[col].copy()
                df[col] = df[col].apply(_to_e164)
                changed = int((df[col] != original.astype(str)).sum())
            else:
                return df, "", f"Unknown format '{fmt}'"
            return df, f"Normalized {changed} value(s) in '{col}' to format='{fmt}'.", None

        # ---------------------------------------------------------------
        elif op == "clip_outliers":
            lower = params.get("lower")
            upper = params.get("upper")
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            clipped = numeric_col.clip(lower=lower, upper=upper)
            changed = int((clipped != numeric_col).sum())
            df[col] = clipped
            return df, f"Clipped {changed} value(s) in '{col}' to [{lower}, {upper}].", None

        # ---------------------------------------------------------------
        elif op == "rename_column":
            new_name = params.get("new_name")
            if not new_name:
                return df, "", "params.new_name is required for rename_column"
            df = df.rename(columns={col: new_name})
            return df, f"Renamed column '{col}' → '{new_name}'.", None

        # ---------------------------------------------------------------
        elif op == "drop_column":
            df = df.drop(columns=[col])
            return df, f"Dropped column '{col}'.", None

        # ---------------------------------------------------------------
        elif op == "fix_inconsistency":
            # Generic: fill missing from computed value
            # For hard task: compute net_amount = gross_amount - tax
            if col == "net_amount":
                if all(c in df.columns for c in ["gross_amount", "tax"]):
                    mask = df["net_amount"].isna()
                    df.loc[mask, "net_amount"] = (
                        pd.to_numeric(df.loc[mask, "gross_amount"], errors="coerce")
                        - pd.to_numeric(df.loc[mask, "tax"], errors="coerce")
                    ).round(2)
                    filled = int(mask.sum())
                    return df, f"Computed {filled} missing net_amount value(s) from gross_amount - tax.", None
            if col == "total":
                if all(c in df.columns for c in ["debit", "credit"]):
                    df["total"] = (
                        pd.to_numeric(df["debit"], errors="coerce")
                        + pd.to_numeric(df["credit"], errors="coerce")
                    ).round(2)
                    return df, "Recomputed total = debit + credit for all rows.", None
            if col == "currency_raw":
                def _normalize_currency(v: Any) -> str:
                    if pd.isna(v):
                        return v
                    num = re.sub(r"[^\d.]", "", str(v).replace("USD", "").strip())
                    try:
                        return f"${float(num):.2f}"
                    except Exception:
                        return str(v)
                df[col] = df[col].apply(_normalize_currency)
                return df, f"Normalized currency format in '{col}'.", None
            # Fallback
            return df, f"fix_inconsistency on '{col}' — no specific rule matched.", None

        else:
            return df, "", f"Unknown operation '{op}'"

    except Exception as exc:
        return df, "", str(exc)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DataCleanEnvironment:
    """
    OpenEnv-compatible environment for data cleaning tasks.

    Attributes exposed for the framework:
      reward : float
      done   : bool
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_id: str = ""
        self._step_count: int = 0
        self._reward: float = 0.0
        self._done: bool = False

        # DataFrames
        self._current_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._dirty_df: Optional[pd.DataFrame] = None  # initial dirty snapshot

        # Grading context
        self._original_null_counts: Dict[str, int] = {}
        self._original_dup_count: int = 0
        self._prev_quality: float = 0.0
        self._current_quality: float = 0.0

        # Action history
        self._actions_taken: List[str] = []

    # ------------------------------------------------------------------
    # Properties required by the framework
    # ------------------------------------------------------------------

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    async def reset_async(self, task_id: str = "easy", seed: Optional[int] = None, **kwargs) -> DataCleanObservation:
        return self.reset(task_id=task_id, seed=seed, **kwargs)

    async def step_async(self, action: DataCleanAction) -> DataCleanObservation:
        return self.step(action)

    def close(self) -> None:
        pass

    def reset(self, task_id: str = "easy", seed: Optional[int] = None, **kwargs) -> DataCleanObservation:
        """Start a new episode."""
        if task_id not in ("easy", "medium", "hard"):
            task_id = "easy"

        # Pick a seed: caller can override, otherwise use variant rotation
        if seed is None:
            seed = _SEED_VARIANTS[0]

        self._episode_id = str(uuid.uuid4())[:8]
        self._task_id = task_id
        self._step_count = 0
        self._reward = 0.0
        self._done = False
        self._actions_taken = []

        dirty_df, clean_df = generate_task_data(task_id, seed=seed)
        self._clean_df = clean_df.reset_index(drop=True)
        self._dirty_df = dirty_df.copy()
        self._current_df = dirty_df.copy()

        # Snapshot original quality context
        self._original_null_counts = {
            col: int(dirty_df[col].isna().sum()) for col in dirty_df.columns
        }
        self._original_dup_count = int(dirty_df.duplicated().sum())

        # Compute initial quality
        grade = grade_submission(
            self._current_df, self._clean_df, task_id,
            self._original_null_counts, self._original_dup_count
        )
        self._prev_quality = grade["score"]
        self._current_quality = grade["score"]

        print(f"[DEBUG] reset complete task={task_id} df shape: {self._current_df.shape}", flush=True)
        return self._build_observation(
            last_action_result="Episode started. Analyze the data and begin cleaning.",
            last_action_error=None,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: DataCleanAction) -> DataCleanObservation:
        """Execute one cleaning action and return updated observation."""
        try:
            return self._step_inner(action)
        except Exception:
            import traceback
            traceback.print_exc()
            self._reward = -0.02
            return self._build_observation(
                last_action_result="",
                last_action_error="Internal error during step — see server logs.",
            )

    def _step_inner(self, action: DataCleanAction) -> DataCleanObservation:
        if self._current_df is None:
            self.reset()

        if self._done:
            return self._build_observation(
                last_action_result="Episode already finished.",
                last_action_error="Episode is done. Call reset() to start a new episode.",
            )

        self._step_count += 1
        self._prev_quality = self._current_quality

        if action.operation == "submit":
            # Final grading
            self._done = True
            grade = grade_submission(
                self._current_df, self._clean_df, self._task_id,
                self._original_null_counts, self._original_dup_count,
            )
            self._current_quality = grade["score"]
            self._reward = self._current_quality  # final reward = absolute score
            self._actions_taken.append("submit()")
            return self._build_observation(
                last_action_result=f"Submitted. Final score: {self._current_quality:.3f}. {grade['details']}",
                last_action_error=None,
            )

        # Apply action
        new_df, result_msg, error_msg = _execute_action(self._current_df, action)

        if error_msg:
            # Invalid action — no change, small penalty
            self._reward = -0.02
            action_summary = f"{action.operation}({action.column or ''}) [ERROR]"
            self._actions_taken.append(action_summary)
        else:
            self._current_df = new_df

            # Recompute quality
            grade = grade_submission(
                self._current_df, self._clean_df, self._task_id,
                self._original_null_counts, self._original_dup_count,
            )
            self._current_quality = grade["score"]
            delta = self._current_quality - self._prev_quality

            if delta > 0.001:
                self._reward = delta  # positive improvement
            elif delta < -0.001:
                self._reward = -0.05  # destructive action
            else:
                self._reward = -0.01  # no-op

            action_summary = f"{action.operation}({action.column or ''})"
            self._actions_taken.append(action_summary)

        # End episode if max steps reached
        if self._step_count >= MAX_STEPS:
            self._done = True
            if error_msg is None:
                result_msg += f" [Max steps reached. Final score: {self._current_quality:.3f}]"

        return self._build_observation(
            last_action_result=result_msg,
            last_action_error=error_msg,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> DataCleanState:
        return DataCleanState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            current_score=round(self._current_quality, 6),
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_action_result: str,
        last_action_error: Optional[str],
    ) -> DataCleanObservation:
        df = self._current_df
        if df is None:
            df = pd.DataFrame()

        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
        dup_count = int(df.duplicated().sum())
        issues = compute_quality_issues(df, self._task_id)

        return DataCleanObservation(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS.get(self._task_id, ""),
            difficulty=self._task_id,
            data_preview=_df_preview(df),
            columns=list(df.columns),
            shape=[len(df), len(df.columns)],
            dtypes={col: str(df[col].dtype) for col in df.columns},
            null_counts=null_counts,
            duplicate_count=dup_count,
            quality_score=round(self._current_quality, 6),
            issues_found=issues,
            issues_remaining=len(issues),
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            actions_taken=list(self._actions_taken[-10:]),
        )
