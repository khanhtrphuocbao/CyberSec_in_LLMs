# Custom-question agent trace implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Agent Trace render the most recently run custom question while following Question Runner's existing source choice.

**Architecture:** `demo/app.py` will retain each custom `SolveResult.to_dict()` under `st.session_state["custom_run_results"]`. Agent Trace will read `st.session_state["runner_question_source"]`: `Test set` preserves artifact lookup, while `Câu hỏi tuỳ ý` reads that in-session mapping. The existing trace panels remain shared by one renderer that accepts a result dictionary.

**Tech Stack:** Python 3.14, Streamlit session state, existing MedQA result JSON.

## Global constraints

- Do not write custom question results into `results/V*/`, `results/single_question/`, or benchmark checkpoints.
- Preserve V0–V4 solve semantics, prompts, and CLI behaviour.
- The trace source must be the existing `runner_question_source` control; do not add a second source chooser.
- Tests must use Streamlit AppTest or fake result dictionaries only; never call OpenAI, Chroma, or a CLI runner.

---

### Task 1: Add a regression test for source-following custom traces

**Files:**
- Modify: `tests/test_demo_app.py`
- Modify: `demo/app.py`

**Interfaces:**
- Consumes `st.session_state["runner_question_source"]` and `st.session_state["custom_run_results"]`.
- Produces the existing Agent Trace panels for a selected custom variant or an empty-state message when no result exists.

- [x] **Step 1: Write the failing AppTest**

```python
app = AppTest.from_file(str(REPOSITORY_ROOT / "demo" / "app.py")).run(timeout=60)
app.segmented_control[0].set_value("Câu hỏi tuỳ ý").run(timeout=60)
app.session_state["custom_run_results"] = {
    "V3": {"predicted_answer": "B", "confidence": 0.9, "metadata": {"planner_trace": {"steps": []}}}
}
app.selectbox(key="trace_variant").set_value("V3").run(timeout=60)
self.assertIn("B", [metric.value for metric in app.metric])
```

- [x] **Step 2: Run the focused test to verify RED**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_app.py' -v`

Expected: the test fails because Agent Trace only uses a test-set artifact.

- [x] **Step 3: Implement custom trace resolution and shared rendering**

```python
if st.session_state.get("runner_question_source") == "Câu hỏi tuỳ ý":
    result = st.session_state.get("custom_run_results", {}).get(variant)
else:
    result = select_question_result(variant, question_id, RESULTS_ROOT, SINGLE_ROOT)
```

Keep metric and RAG/Plan/Examiner/Evaluator/Raw JSON panel rendering identical for both result sources. Only attempt to load a CLI log for test-set artifacts.

- [x] **Step 4: Run focused tests to verify GREEN**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -p 'test_demo_*.py' -v`

Expected: all dashboard tests pass without API calls.

- [x] **Step 5: Run the full unit suite**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -v`

Expected: all repository unit tests pass.
