"""
FastAPI application for data_clean_env.

Uses plain FastAPI HTTP endpoints for all routes — no WebSocket, no create_app().
This ensures /reset, /step, /state all work as standard HTTP POST/GET.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import DataCleanAction, DataCleanObservation, DataCleanState
from server.environment import DataCleanEnvironment

# ---------------------------------------------------------------------------
# App + single global environment instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="data_clean_env",
    description="Data Cleaning & Transformation OpenEnv Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = DataCleanEnvironment()

# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "data_clean_env"}


@app.post("/reset")
def reset(body: Optional[Dict[str, Any]] = None):
    kwargs = body or {}
    task_id = kwargs.get("task_id", "easy")
    seed = kwargs.get("seed", None)
    try:
        obs = _env.reset(task_id=task_id, seed=seed)
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
def step(body: Dict[str, Any]):
    try:
        action = DataCleanAction(**body)
        obs = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": round(_env.reward, 6),
            "done": _env.done,
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
def state():
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    from server.data_generator import TASK_DESCRIPTIONS
    return {
        "tasks": [
            {"task_id": tid, "difficulty": tid, "description": desc}
            for tid, desc in TASK_DESCRIPTIONS.items()
        ]
    }


@app.post("/grader")
def run_grader(body: Optional[Dict[str, Any]] = None):
    from server.grader import grade_submission
    body = body or {}
    task_id = body.get("task_id") or _env._task_id or "easy"
    if _env._current_df is None or _env._clean_df is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    result = grade_submission(
        _env._current_df,
        _env._clean_df,
        task_id,
        _env._original_null_counts,
        _env._original_dup_count,
    )
    return result


@app.get("/baseline")
def run_baseline():
    from server.data_generator import generate_task_data
    from server.grader import grade_submission
    results = {}
    for task_id in ("easy", "medium", "hard"):
        dirty_df, clean_df = generate_task_data(task_id, seed=42)
        dirty_grade = grade_submission(dirty_df, clean_df, task_id)
        perfect_grade = grade_submission(clean_df, clean_df, task_id)
        results[task_id] = {
            "dirty_score": round(dirty_grade["score"], 4),
            "perfect_score": round(perfect_grade["score"], 4),
        }
    return {"baseline_scores": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
