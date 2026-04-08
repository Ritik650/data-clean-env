"""
Baseline inference script for data_clean_env.

Runs an LLM agent against all 3 tasks (easy, medium, hard) using plain
HTTP requests — no Docker required.

Required environment variables:
  HF_TOKEN      — Hugging Face / API key
  API_BASE_URL  — LLM API base URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    — Model to use      (default: Qwen/Qwen2.5-72B-Instruct)
  SPACE_URL     — Environment URL   (default: https://Ritik1825-data-clean-env.hf.space)
  IMAGE_NAME    — Docker image name (optional, unused in HTTP mode)

Stdout format (MANDATORY — do not change):
  [START] task=<task_id> env=<bench_name> model=<model>
  [STEP]  step=<n> action=<act> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME: str = os.getenv("IMAGE_NAME", "data-clean-env:latest")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL: str = os.getenv("SPACE_URL", "https://Ritik1825-data-clean-env.hf.space").rstrip("/")

BENCHMARK: str = "data_clean_env"
MAX_STEPS: int = 15
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 512
REQUEST_TIMEOUT: int = 60

# ---------------------------------------------------------------------------
# HTTP helpers (no Docker needed)
# ---------------------------------------------------------------------------

def _post(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{SPACE_URL}{endpoint}", json=data, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def env_reset(task_id: str) -> Dict[str, Any]:
    return _post("/reset", {"task_id": task_id})


def env_step(operation: str, column: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/step", {"operation": operation, "column": column, "params": params})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data cleaning agent. You receive a dirty dataset and must fix it step by step.

Available operations:
- fill_missing      Fill null values.          params: {strategy: "mean"|"median"|"mode"|"value", value: <val>}
- drop_rows         Remove rows.               params: {condition: "duplicates"|"nulls"|"expression", expression: <expr>}
- deduplicate       Remove duplicate rows.     params: {}
- replace_value     Replace values in column.  params: {old: <val>, new: <val>}
- cast_type         Change column dtype.        params: {dtype: "int"|"float"|"str"|"datetime"}
- normalize_format  Standardize format.        params: {format: "uppercase"|"lowercase"|"titlecase"|"date_iso"|"phone_e164"}
- clip_outliers     Clip extreme values.        params: {lower: <num>, upper: <num>}
- fix_inconsistency Fix cross-col logic.        params: {} (for net_amount, total, currency_raw)
- drop_column       Remove a column.            params: {}
- submit            Submit for final grading.   params: {}

Always respond with ONLY a valid JSON object — no markdown, no explanations:
{"operation": "...", "column": "...", "params": {...}}

Strategy:
1. Fix structural issues first (remove duplicates, fix data types)
2. Handle missing values (fill or drop)
3. Normalize formats (dates, phones, casing)
4. Fix cross-column inconsistencies
5. Submit when clean or at step 12+
"""


def _make_user_prompt(obs: Dict[str, Any]) -> str:
    return f"""Current data state:
Task: {obs.get('task_id')} ({obs.get('difficulty')})
Step: {obs.get('step_number')}/{obs.get('max_steps')}
Quality Score: {obs.get('quality_score', 0):.3f}

Data Preview (first 10 rows):
{obs.get('data_preview', '')}

Column Types: {json.dumps(obs.get('dtypes', {}), indent=2)}
Null Counts:  {json.dumps(obs.get('null_counts', {}))}
Duplicates:   {obs.get('duplicate_count', 0)}
Shape:        {obs.get('shape', [0, 0])[0]} rows x {obs.get('shape', [0, 0])[1]} cols

Issues Remaining ({obs.get('issues_remaining', 0)}):
{json.dumps(obs.get('issues_found', [])[:8], indent=2)}

Last Action Result: {obs.get('last_action_result', '')}
Last Action Error:  {obs.get('last_action_error')}
Actions Taken So Far: {json.dumps(obs.get('actions_taken', [])[-6:])}

What cleaning operation should be performed next?
Respond with ONLY a JSON object."""


# ---------------------------------------------------------------------------
# Logging — MANDATORY FORMAT
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, obs: Dict[str, Any]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _make_user_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"operation": "submit", "column": null, "params": {}}'


def parse_action(text: str) -> Dict[str, Any]:
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            inner, in_block = [], False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True; continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            text = "\n".join(inner).strip()
        data = json.loads(text)
        return {
            "operation": data.get("operation", "submit"),
            "column": data.get("column") or None,
            "params": data.get("params") or {},
        }
    except Exception as exc:
        print(f"[DEBUG] Action parse error: {exc} | raw='{text[:200]}'", flush=True)
        return {"operation": "submit", "column": None, "params": {}}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> None:
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        data = env_reset(task_id)
        obs = data.get("observation", data)
        done = bool(data.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Auto-submit near step limit
            if obs.get("step_number", 0) >= MAX_STEPS - 1:
                action = {"operation": "submit", "column": None, "params": {}}
            else:
                model_response = get_model_action(client, obs)
                action = parse_action(model_response)

            result = env_step(action["operation"], action["column"], action["params"])
            obs = result.get("observation", result)
            reward = float(result.get("reward") or 0.0)
            done = bool(result.get("done", False))
            error = obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action['operation']}({action['column'] or ''})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = float(obs.get("quality_score", 0.001))
        score = max(0.001, min(0.999, score))
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Task '{task_id}' failed: {exc}", flush=True)
        score = 0.001
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in ("easy", "medium", "hard"):
        run_task(client, task_id)


if __name__ == "__main__":
    main()
