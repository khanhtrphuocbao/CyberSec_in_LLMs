# Streamlit Benchmark Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit dashboard that compares V0–V4, runs selected test-set indices, and renders agent traces.

**Architecture:** Keep Streamlit UI in `demo/app.py`; move all result parsing and trace selection into pure functions in `demo/data.py`; use `demo/runner.py` solely to construct and execute existing CLI wrappers. The UI reads committed full results and generated single-question artifacts, never benchmark checkpoints.

**Tech Stack:** Python 3.14, Streamlit, pandas, Plotly, existing `MedQALoader` and `run_v*.py` wrappers.

## Global Constraints

- Dashboard reads `results/V*/results_V*.json`; no dashboard page may modify full benchmark artifacts.
- API credentials stay in `.env` and are never rendered, logged, or committed.
- Run V0–V4 sequentially via existing CLI wrappers; do not alter their prompt or solve semantics.
- Tests use fixture JSON and fake subprocess runners only; no network, Chroma, or OpenAI calls.
- Historical cumulative token telemetry must be labelled non-comparable, not averaged as a true per-question cost.

---

### Task 1: Result and trace data layer

**Files:**
- Create: `demo/__init__.py`, `demo/data.py`
- Create: `tests/test_demo_data.py`

**Interfaces:**
- Produces `load_variant_results(results_root: Path) -> dict[str, list[dict]]`.
- Produces `variant_summary(rows: list[dict]) -> dict[str, float | int]`.
- Produces `select_question_result(variant, question_id, results_root, single_root) -> dict | None`.

- [ ] **Step 1: Write failing tests**

```python
def test_variant_summary_counts_valid_correct_rows_only():
    summary = variant_summary([
        {"is_valid": True, "is_correct": True, "confidence": 0.9, "latency_seconds": 2.0},
        {"is_valid": False, "is_correct": False, "confidence": 0.0, "latency_seconds": 3.0},
    ])
    self.assertEqual(summary["valid"], 1)
    self.assertEqual(summary["correct"], 1)
    self.assertEqual(summary["accuracy"], 0.5)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_data.py' -v`

Expected: import failure because `demo.data` does not exist.

- [ ] **Step 3: Implement the pure result functions**

```python
def load_variant_results(results_root: Path) -> dict[str, list[dict]]:
    return {variant: json.loads(path.read_text()) for variant, path in discovered_paths.items()}
```

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_data.py' -v`

Expected: PASS.

### Task 2: Existing-CLI runner adapter

**Files:**
- Create: `demo/runner.py`
- Create: `tests/test_demo_runner.py`

**Interfaces:**
- Produces `build_variant_command(python, repository_root, variant, question_index, top_k, two_step) -> list[str]`.
- Produces `run_variant(...) -> subprocess.CompletedProcess[str]`.

- [ ] **Step 1: Write failing command-construction test**

```python
command = build_variant_command("python", Path("/repo"), "V3", 10, 5, True)
self.assertEqual(command, ["python", "run_v3.py", "--question-index", "10", "--top-k", "5", "--two-step-retrieval"])
```

- [ ] **Step 2: Run targeted test and verify RED**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_runner.py' -v`

Expected: import failure because `demo.runner` does not exist.

- [ ] **Step 3: Implement command builder and subprocess adapter**

```python
return subprocess.run(command, cwd=repository_root, env=environment, capture_output=True, text=True)
```

- [ ] **Step 4: Run targeted test and verify GREEN**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_runner.py' -v`

Expected: PASS.

### Task 3: Streamlit app and local setup

**Files:**
- Create: `demo/app.py`, `demo/requirements.txt`
- Modify: `STRUCTURE.md`
- Test: `tests/test_demo_data.py`, `tests/test_demo_runner.py`

**Interfaces:**
- Consumes `demo.data` for all file parsing and `demo.runner` for API-triggering CLI calls.
- Provides `streamlit run demo/app.py` as the local entry point.

- [ ] **Step 1: Add a failing import smoke test**

```python
completed = subprocess.run([sys.executable, "-m", "py_compile", "demo/app.py"], capture_output=True, text=True)
self.assertEqual(completed.returncode, 0, completed.stderr)
```

- [ ] **Step 2: Implement the three tabs**

```python
dashboard_tab, runner_tab, trace_tab = st.tabs(["Benchmark dashboard", "Question runner", "Agent trace"])
```

- [ ] **Step 3: Install demo dependencies and run verification**

Run: `../venv/bin/pip install -r demo/requirements.txt && PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -v`

Expected: dependencies install and full suite pass.

- [ ] **Step 4: Manual local smoke test**

Run: `../venv/bin/streamlit run demo/app.py --server.headless true --server.port 8501`

Expected: server starts and Dashboard renders committed V0–V4 result cards without an API request.
