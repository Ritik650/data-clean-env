"""
Pydantic models for data_clean_env.
Action, Observation, and State models following the OpenEnv spec.
"""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Base classes — mirrored locally so the package can work even if openenv-core
# is not installed (e.g. during local testing without the full SDK).
# When openenv-core IS installed the proper base classes are imported below.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State  # type: ignore
except ImportError:  # pragma: no cover
    class Action(BaseModel):  # type: ignore
        pass

    class Observation(BaseModel):  # type: ignore
        pass

    class State(BaseModel):  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DataCleanAction(Action):
    """Action the agent takes to clean the data."""

    operation: str = Field(
        ...,
        description=(
            "Cleaning operation: 'fill_missing', 'drop_rows', 'replace_value', "
            "'cast_type', 'deduplicate', 'normalize_format', 'fix_inconsistency', "
            "'clip_outliers', 'rename_column', 'drop_column', 'submit'"
        ),
    )
    column: Optional[str] = Field(None, description="Target column name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Operation parameters. Examples: "
            "fill_missing: {'strategy': 'mean'|'median'|'mode'|'value', 'value': <val>}, "
            "replace_value: {'old': <val>, 'new': <val>}, "
            "cast_type: {'dtype': 'int'|'float'|'str'|'datetime'}, "
            "clip_outliers: {'lower': <num>, 'upper': <num>}, "
            "normalize_format: {'format': 'uppercase'|'lowercase'|'titlecase'|'date_iso'|'phone_e164'}, "
            "drop_rows: {'condition': 'duplicates'|'nulls'|'expression', 'expression': <expr>}"
        ),
    )

    @field_validator('params', mode='before')
    @classmethod
    def parse_params(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v) if v.strip() else {}
            except json.JSONDecodeError:
                return {}
        return v if v is not None else {}

    @field_validator('column', mode='before')
    @classmethod
    def parse_column(cls, v):
        if isinstance(v, str) and v.strip() == '':
            return None
        return v


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class DataCleanObservation(Observation):
    """What the agent sees after each step."""

    task_id: str = Field(..., description="Task identifier: 'easy', 'medium', 'hard'")
    task_description: str = Field(..., description="Natural language description of what needs cleaning")
    difficulty: str = Field(..., description="easy | medium | hard")

    # Data snapshot
    data_preview: str = Field(..., description="First 10 rows of current data as formatted string")
    columns: List[str] = Field(..., description="Column names")
    shape: List[int] = Field(..., description="[num_rows, num_cols]")
    dtypes: Dict[str, str] = Field(..., description="Column name -> dtype mapping")

    # Quality report
    null_counts: Dict[str, int] = Field(..., description="Null count per column")
    duplicate_count: int = Field(0, description="Number of duplicate rows")
    quality_score: float = Field(0.0, description="Current overall quality score 0.0-1.0")
    issues_found: List[str] = Field(default_factory=list, description="List of detected issues")
    issues_remaining: int = Field(0, description="Number of issues still to fix")

    # Step info
    step_number: int = Field(0)
    max_steps: int = Field(15)
    last_action_result: str = Field("", description="Result/feedback from last action")
    last_action_error: Optional[str] = Field(None, description="Error from last action if any")

    # History
    actions_taken: List[str] = Field(
        default_factory=list, description="Summary of actions taken so far"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DataCleanState(State):
    """Internal episode metadata exposed via /state."""

    episode_id: str = ""
    task_id: str = ""
    step_count: int = 0
    current_score: float = 0.0
    done: bool = False
