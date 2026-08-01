# Streamlit benchmark dashboard design

## Goal

Provide a local Streamlit demo for presenting the V0–V4 ablation study, inspecting one MedQA question, and running selected variants by zero-based test-set index without changing benchmark semantics.

## Audience and visual direction

The audience is the project team and a midterm-review panel. The application’s single job is to make evidence visible: compare variants globally, then drill down into one trace. The visual language is a clinical evidence board: paper background `#F4F7F8`, ink `#102A43`, retrieval teal `#0F766E`, verifier amber `#B45309`, action blue `#0077B6`, and error red `#B42318`. Serif headings and compact monospaced metric labels distinguish interpretation from telemetry. The signature element is a horizontal agent evidence ribbon: RAG → Planner → Examiner → Evaluator, populated only when the selected result contains that stage.

## Architecture

`demo/data.py` is a dependency-free data layer. It discovers committed result files below `results/V*/`, normalizes per-question rows, computes accuracy/validity/confidence/latency summaries, and extracts trace panels without executing LLM calls.

`demo/runner.py` builds a safe subprocess command for the existing `run_v0.py`…`run_v4.py` wrappers. The Streamlit page runs selected variants sequentially, preserving the current `OPENAI_API_KEY`/`.env` behaviour and consuming the JSON/log artifacts already produced by the CLI.

`demo/app.py` contains only Streamlit presentation logic. It has Dashboard, Question Runner, and Agent Trace tabs and uses the data/runner modules. The app never writes benchmark `results_V*/results_V*.json` files.

## Behaviour

- Result discovery reads `results/V0/results_V0.json` through `results/V4/results_V4.json`; missing files produce an actionable empty state.
- Dashboard shows accuracy, validity rate, confidence, and result-derived latency. It labels historical token metrics as non-comparable when a summary file reports cumulative telemetry.
- Question Runner lets a user choose an existing dataset index from 0 to `len(test_set)-1`, inspect its question/options, and run one or more variants sequentially through the existing CLI. It forwards `top_k` and the two-step toggle only; API keys remain in `.env`.
- Agent Trace displays the best available result for the selected question and variant, preferring `results/single_question/<variant>/<question_id>.json`, then falling back to the committed full benchmark row. It presents RAG, plan, examiner analysis, evaluator history, raw JSON, and the matching log if present.

### Custom-question trace extension

- Agent Trace follows the `Nguồn câu hỏi` choice made in Question Runner; it does not introduce a second source selector.
- When that choice is `Câu hỏi tuỳ ý`, the tab reads the in-session result for the selected trace variant and renders the same RAG, planner, examiner, evaluator, metric, and raw-JSON panels.
- The custom trace is held in `st.session_state` only. It is intentionally absent after a browser-session/server restart and is never mixed into benchmark or single-question artifact folders.
- If the selected variant was not run for the current custom question, Agent Trace displays an actionable empty state.

## Safety and scope

- The dashboard is local-only; it does not expose API keys or add authentication/network services.
- It does not amend prompts, RAG configuration defaults, answer selection, or V0–V4 workflow logic.
- The app runs one selected variant at a time in a subprocess; selecting multiple variants runs them sequentially to avoid rate-limit spikes.
- Generated single-question outputs remain under existing ignored paths.

## Dependencies and validation

Add `demo/requirements.txt` with `streamlit`, `pandas`, and `plotly`; use the existing project venv. Unit tests cover data normalisation, trace selection, and CLI command construction with fakes; no unit test may call OpenAI, Chroma, or Streamlit. A final manual smoke test starts Streamlit and loads committed results without API calls.
