---
title: data_clean_env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# data_clean_env

**Data Cleaning & Transformation Environment for OpenEnv**

An OpenEnv-compatible reinforcement learning environment where AI agents learn to clean dirty tabular datasets through iterative `step()` / `reset()` / `state()` interactions.

---

## Motivation

Data cleaning is the **#1 real-world bottleneck in data science** — analysts spend 60–80% of their time on it. This environment:

- Provides **immediate, measurable value** — every data pipeline in every company deals with dirty data
- Offers **graded, deterministic rewards** — 4-component scoring (completeness, accuracy, consistency, structure)
- Has **natural difficulty progression** — easy → medium → hard tasks with increasingly subtle, interdependent issues
- Generates **dense per-step rewards** — no sparse end-of-episode signal; agents learn incrementally

---

## Action Space

All actions are `DataCleanAction` objects:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | Cleaning operation name (see table below) |
| `column` | `str \| null` | Target column (required for most operations) |
| `params` | `dict` | Operation-specific parameters |

### Operations

| Operation | Description | Key Params |
|-----------|-------------|------------|
| `fill_missing` | Fill null values in a column | `strategy`: `mean`/`median`/`mode`/`value`; `value`: fallback |
| `drop_rows` | Remove rows by condition | `condition`: `duplicates`/`nulls`/`expression`; `expression` |
| `deduplicate` | Remove all exact duplicate rows | — |
| `replace_value` | Replace specific values in a column | `old`, `new` |
| `cast_type` | Change column data type | `dtype`: `int`/`float`/`str`/`datetime` |
| `normalize_format` | Standardize string/date/phone formats | `format`: `uppercase`/`lowercase`/`titlecase`/`date_iso`/`phone_e164` |
| `clip_outliers` | Clip numeric values to a range | `lower`, `upper` |
| `fix_inconsistency` | Fix cross-column logic errors | column-dependent |
| `rename_column` | Rename a column | `new_name` |
| `drop_column` | Remove a column entirely | — |
| `submit` | Submit current state for final grading | — |

---

## Observation Space

Each step returns a `DataCleanObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | `easy` / `medium` / `hard` |
| `task_description` | `str` | Natural-language description of the cleaning task |
| `difficulty` | `str` | Same as `task_id` |
| `data_preview` | `str` | First 10 rows formatted as a table |
| `columns` | `List[str]` | Column names |
| `shape` | `List[int]` | `[num_rows, num_cols]` |
| `dtypes` | `Dict[str, str]` | Column → dtype |
| `null_counts` | `Dict[str, int]` | Null count per column |
| `duplicate_count` | `int` | Number of duplicate rows |
| `quality_score` | `float` | Current overall quality 0.0–1.0 |
| `issues_found` | `List[str]` | Human-readable list of detected issues |
| `issues_remaining` | `int` | Number of remaining issues |
| `step_number` | `int` | Current step |
| `max_steps` | `int` | 15 |
| `last_action_result` | `str` | Feedback from the last action |
| `last_action_error` | `str \| null` | Error message if action was invalid |
| `actions_taken` | `List[str]` | Last 10 actions as summary strings |

---

## Tasks

### Task 1 — Easy: "Clean Employee Records"

50-row employee dataset with 5–7 obvious issues:

| Issue | Fix |
|-------|-----|
| `salary` stored as `"$45000.00"` (string with `$` prefix) | `cast_type` → `float` |
| `department` inconsistent casing: `"engineering"`, `"ENGINEERING"` | `normalize_format` → `titlecase` |
| `salary` has 3 missing values | `fill_missing` → `median` |
| `department` has 2 missing values | `fill_missing` → `mode` |
| 3 exact duplicate rows | `deduplicate` |

**Expected score:** 0.65–0.85

---

### Task 2 — Medium: "Fix Customer Orders Dataset"

100-row orders dataset with 8–12 more subtle issues:

| Issue | Fix |
|-------|-----|
| `order_date` in mixed formats (`MM/DD/YYYY`, `DD-Mon-YY`, ISO) | `normalize_format` → `date_iso` |
| `phone` in inconsistent formats | `normalize_format` → `phone_e164` |
| `email` has 5 null values | `drop_rows` → `nulls` |
| `order_total` has negative values | `clip_outliers` lower=0 |
| `status` has typos: `"shiped"`, `"cancled"` | `replace_value` |
| `quantity` has string values: `"two"`, `"three"` | `replace_value` + `cast_type` → `int` |
| `order_total` has 2 extreme outliers ($999,999) | `clip_outliers` upper=4000 |

**Expected score:** 0.35–0.55

---

### Task 3 — Hard: "Reconcile Financial Transactions"

150-row financial dataset with 15–20 complex, interdependent issues:

| Issue | Fix |
|-------|-----|
| ALL rows: `debit + credit ≠ total` | `fix_inconsistency` on `total` |
| ALL rows: `settlement_date < tx_date` | `fix_inconsistency` / `replace_value` |
| ALL rows: `currency_raw` in wrong format (`USD X.XX` / plain float) | `fix_inconsistency` on `currency_raw` |
| ALL rows: `category = "Unknown"` (invalid) | `replace_value` |
| `net_amount` missing (80 rows) | `fix_inconsistency` on `net_amount` |
| `tax` missing (60 rows) | `fill_missing` → `value` |
| `debit` missing (40 rows) | `fill_missing` → `value` |
| Orphan `account_id` values (30 rows) | `drop_rows` → `expression` |
| 20 exact duplicate rows | `deduplicate` |

**Expected initial score:** 0.20–0.30

---

## Reward Function

### Per-step reward
```
Δ = new_quality_score − old_quality_score
step_reward = Δ           if Δ > 0.001   (improvement)
step_reward = −0.05       if Δ < −0.001  (destructive action)
step_reward = −0.01       if |Δ| ≤ 0.001 (no-op)
step_reward = −0.02       on invalid action (column not found, etc.)
```

### Final reward (on submit or max_steps)
```
final_reward = quality_score ∈ [0.0, 1.0]
```

### Quality score decomposition
```
quality_score = 0.25 × completeness
              + 0.25 × accuracy
              + 0.25 × consistency
              + 0.25 × structure
```

| Component | Measures |
|-----------|----------|
| **Completeness** | % of originally-null cells that are now non-null |
| **Accuracy** | Cell-by-cell match against ground truth (1% tolerance for floats) |
| **Consistency** | Format compliance: ISO dates, E.164 phones, valid categories, etc. |
| **Structure** | Correct row count, no duplicates, correct column dtypes |

---

## Setup

### Local (no Docker)

```bash
cd data_clean_env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash

docker build -t data-clean-env .
docker run -p 7860:7860 data-clean-env
```

### Test the endpoints

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"operation": "deduplicate", "column": null, "params": {}}'

# Check state
curl http://localhost:7860/state

# Run baseline score check
curl http://localhost:7860/baseline
```

---

## Running the Baseline Inference Script

```bash
export IMAGE_NAME="data-clean-env:latest"
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

Expected stdout:
```
[START] task=easy env=data_clean_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=deduplicate() reward=0.05 done=false error=null
[STEP] step=2 action=cast_type(salary) reward=0.12 done=false error=null
...
[END] success=true steps=8 score=0.742 rewards=0.05,0.12,...

[START] task=medium env=data_clean_env model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=15 score=0.421 rewards=...

[START] task=hard env=data_clean_env model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=15 score=0.187 rewards=...
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "easy\|medium\|hard", "seed": 42}` |
| `POST` | `/step` | Take an action. Body: `DataCleanAction` JSON |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/grader` | Run grader on current state |
| `GET` | `/baseline` | Run heuristic baseline, return dirty vs clean scores |

---

## Baseline Scores (Grader Validation)

The `/baseline` endpoint verifies the grader produces differentiated scores:

| Task | Initial Score (dirty) | After Cleaning | Baseline Agent | Perfect Score |
|------|-----------------------|----------------|----------------|---------------|
| easy | ~0.65 | ~0.99 | ~0.85 | 1.00 |
| medium | ~0.55 | ~0.80 | ~0.50 | 1.00 |
| hard | ~0.25 | ~0.55 | ~0.30 | 1.00 |

- **Initial Score**: quality of the unmodified dirty dataset
- **After Cleaning**: score achievable by applying all correct operations
- **Baseline Agent**: score achieved by the Qwen2.5-72B heuristic agent in `inference.py`
- **Perfect Score**: score when submitting the ground-truth clean dataset

The gap between initial and perfect score is the agent's learning signal.

---

## Project Structure

```
data_clean_env/
├── __init__.py               # Package exports
├── models.py                 # Pydantic Action, Observation, State models
├── client.py                 # DataCleanEnv EnvClient subclass
├── inference.py              # Baseline inference script (root-level, MANDATORY)
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
├── Dockerfile                # Container definition
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI application
    ├── environment.py        # Core DataCleanEnvironment class
    ├── data_generator.py     # Deterministic dirty/clean dataset generation
    └── grader.py             # Deterministic 4-component grading logic
```

---

## HF Spaces Deployment

```bash
# Install openenv CLI
pip install openenv-core

# Validate locally
openenv validate

# Push to HF Spaces
openenv push --repo-id Ritik1825/data-clean-env
```

After deployment, validate the live endpoint:
```bash
curl https://Ritik1825-data-clean-env.hf.space/health
curl -X POST https://Ritik1825-data-clean-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```
