# Single-question CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, one-question CLI execution for V0–V4.

**Architecture:** A common module selects one zero-based dataset index and calls `MedQASystem.solve()`. Thin V0/V1 wrappers call it directly, while V2/V3/V4 retain their benchmark mode unless `--question-index` is supplied.

**Tech Stack:** Python standard library `argparse`, `contextlib`, JSON; existing MedQA loader and system.

## Global Constraints

- Use `OPENAI_API_KEY` as the default API-key environment variable.
- Do not make unit tests call network APIs.
- Do not modify output/checkpoint semantics of full V2–V4 benchmark runs.
- Store one-question outputs below `results/single_question/` and logs below `logs/single_question/`.

---

### Task 1: Common single-question execution module

**Files:**
- Create: `single_question_cli.py`
- Test: `tests/test_single_question_cli.py`

**Interfaces:**
- Produces `build_single_question_parser(variant: str) -> argparse.ArgumentParser`.
- Produces `run_single_question(args, variant, *, loader=None, system_factory=MedQASystem) -> int`.

- [ ] **Step 1: Write the failing test**

```python
exit_code = run_single_question(args, "V3", loader=fake_loader, system_factory=fake_system)
assert exit_code == 0
assert fake_system.calls[0]["question_id"] == "q0010"
assert (result_root / "V3" / "q0010.json").exists()
assert (log_root / "V3" / "q0010.log").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest tests.test_single_question_cli -v`

Expected: FAIL because `single_question_cli` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
question = questions[args.question_index]
result = system.solve(..., variant=variant, top_k=args.top_k)
json.dump(result.to_dict(), result_file, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest tests.test_single_question_cli -v`

Expected: PASS.

### Task 2: Variant wrappers and V2–V4 single-question routing

**Files:**
- Create: `run_v0.py`, `run_v1.py`
- Modify: `run_v2.py`, `run_v3.py`, `run_v4.py`, `evaluation/v2_benchmark.py`
- Test: `tests/test_cli_wrappers.py`

**Interfaces:**
- `run_v0.py` through `run_v4.py` accept `--question-index`.
- Existing V2–V4 no-index invocation retains its benchmark runner.

- [ ] **Step 1: Write the failing test**

```python
for wrapper in ("run_v0.py", "run_v1.py", "run_v2.py", "run_v3.py", "run_v4.py"):
    completed = subprocess.run([sys.executable, wrapper, "--help"], ...)
    self.assertEqual(completed.returncode, 0)
    self.assertIn("--question-index", completed.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest tests.test_cli_wrappers -v`

Expected: FAIL because V0/V1 wrappers do not exist and V2–V4 do not expose the option.

- [ ] **Step 3: Write minimal implementation**

```python
if args.question_index is not None:
    return run_single_question(args, "V2")
return run_full_benchmark(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest tests.test_cli_wrappers -v`

Expected: PASS.

### Task 3: Regression suite and live smoke test

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run the complete unit suite**

Run: `PYTHONPATH=.. ../venv/bin/python -m unittest discover -s tests -v`

Expected: PASS with no network requests.

- [ ] **Step 2: Run live index-10 smoke tests**

Run: `HF_HUB_OFFLINE=1 ../venv/bin/python run_v0.py --question-index 10` (repeat V1–V4)

Expected: each command writes one JSON result and one matching log file.
