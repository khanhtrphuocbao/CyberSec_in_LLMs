# MedQA-RAG — Project Structure & Flow

> Text-only multi-agent adaptation of **MedAgent-Pro** for the **MedQA-USMLE** benchmark with 5 ablation variants (V0-V4) and **Two-step Retrieval**.

---

## ⚠️ Important: `.env` Location

The `.env` file is at **`medqa_rag/.env`** (inside the package directory). Scripts in the project root need it copied to root:

```bash
cp medqa_rag/.env .env
```

---

## 📁 Directory Structure

```
MedAgentPro/                        # Project root
│
├── .env                            # API keys (copied from medqa_rag/.env)
├── venv/                           # Python virtual environment
│
├── resume_v1.py                    # V1 benchmark runner (auto-resume from checkpoint)
├── run_v2.py                       # V2 benchmark runner
├── run_v3.py                       # V3 benchmark runner
├── run_v4.py                       # V4 benchmark runner
│
├── results_V1/                     # V1 results
├── results_V2/                     # V2 results
├── results_V3/                     # V3 results
├── results_V4/                     # V4 results
│
└── medqa_rag/                      # Package root
    ├── __init__.py                 #   Package root exports (v2.0.0)
    ├── config.py                   #   Environment / .env loader
    ├── .env                        #   Local secrets (gitignored)
    │
    ├── core/                       #   ⚙️ System orchestration
    │   └── system.py               #     MedQASystem + 5-variant routing (V0-V4)
    │
    ├── agents/                     #   🤖 Multi-agent components
    │   ├── planner.py              #     MedQA_Planner — reasoning plan generator
    │   ├── examiner.py             #     MedQA_Examiner — plan executor + short-term memory
    │   └── evaluator.py            #     MedQA_Evaluator — reasoning verifier
    │
    ├── rag/                        #   🔍 Retrieval-Augmented Generation
    │   ├── retriever.py            #     MedQA_RAG — ChromaDB + Two-step Retrieval
    │   └── data_loader.py          #     MedQALoader — MedQA dataset utilities
    │
    ├── evaluation/                 #   📊 Benchmark & evaluation
    │   └── runner.py               #     MedQAEvaluator — 5-variant benchmark
    │
    └── scripts/                    #   🛠️ Standalone utilities
        └── ingest_sample_data.py   #     Sample medical-text ingestion
```

---

### Running Benchmarks

This repository is self-contained for the V2/V3 benchmark runners and their
regression tests. From a clone (after creating an environment and configuring
`.env` from `.env.example`), run:

```bash
cd medqa_rag

# Unit tests do not call the API.
PYTHONPATH=.. python -m unittest discover -s tests -v

# V2 / V3 benchmark runners. Results and RAG cache stay in this checkout.
HF_HUB_OFFLINE=1 nohup python -u run_v2.py --workers 3 > run_v2.log 2>&1 &
HF_HUB_OFFLINE=1 nohup python -u run_v3.py --workers 2 > run_v3.log 2>&1 &
```

The wrappers load this checkout as the `medqa_rag` package even if the clone
directory has a different name. A running benchmark launched from the former
parent-level wrapper is unaffected.

### Chạy một câu theo index

Các wrapper `run_v0.py` đến `run_v4.py` đều hỗ trợ chạy một dòng cụ thể của
test set bằng index **zero-based** và mặc định đọc `OPENAI_API_KEY` từ `.env`:

```bash
HF_HUB_OFFLINE=1 python -u run_v0.py --question-index 10
HF_HUB_OFFLINE=1 python -u run_v1.py --question-index 10
HF_HUB_OFFLINE=1 python -u run_v2.py --question-index 10
HF_HUB_OFFLINE=1 python -u run_v3.py --question-index 10
HF_HUB_OFFLINE=1 python -u run_v4.py --question-index 10
```

Chế độ này không dùng checkpoint benchmark. Nó ghi JSON vào
`results/single_question/V*/qXXXX.json` và console log vào
`logs/single_question/V*/qXXXX.log`, nên không ghi đè
`results_V*/results_V*.json`.

### Legacy parent-project commands

```bash
cd /Users/mac/Developers/MedQA_RAG/MedAgentPro

# Copy .env if not already at root
cp medqa_rag/.env .env

# V1 — RAG + LLM (26 min)
nohup venv/bin/python resume_v1.py > run_v1.log 2>&1 &

# V2 — Multi-agent (ETA ~15h)
nohup venv/bin/python run_v2.py > run_v2.log 2>&1 &

# V3 — Full system
nohup venv/bin/python run_v3.py > run_v3.log 2>&1 &

# V4 — V3 no evaluator
nohup venv/bin/python run_v4.py > run_v4.log 2>&1 &
```

Scripts auto-resume from checkpoint if interrupted. Check progress:
```bash
tail -f run_v2.log   # or run_v3.log, etc.
```

---

## 🔁 End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER / CLI                                 │
│         Question text + Dict[A,B,C,D] + Correct answer              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  config.py — load .env (OPENAI_API_KEY, OPENAI_API_BASE)   │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  core/system.py → MedQASystem                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Variant Router: V0 / V1 / V2 / V3 / V4                    │    │
│  │  + Two-step Retrieval toggle (use_two_step_retrieval)      │    │
│  │  + Custom API base (Codex/OpenAI/Azure/etc.)               │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
       │           │            │           │            │
       ▼           ▼            ▼           ▼            ▼
      V0          V1           V2          V3           V4
    Direct      RAG +       Multi-agent   Full        Full
      LLM      Direct LLM   no memory    system     system
                              (cleared)              no verifier
       │           │            │           │            │
       │           │            ▼           ▼            ▼
       │           │      ┌─────────────────────────┐
       │           │      │   rag/retriever.py      │
       │           │      │   ┌───────────────────┐ │
       │           │      │   │ Two-step path:    │ │
       │           │      │   │  1. extract_kw    │ │
       │           │      │   │  2. vector_search │ │
       │           │      │   └───────────────────┘ │
       │           │      └─────────────────────────┘
       │           │            │           │
       │           ▼            ▼           ▼
       │      ┌─────────────────────────────────────┐
       │      │     agents/planner.py               │
       │      │     MedQA_Planner.create_plan()     │
       │      │   → JSON reasoning plan (steps)     │
       │      └─────────────────────────────────────┘
       │                       │
       │                       ▼
       │      ┌─────────────────────────────────────┐
       │      │    agents/examiner.py               │
       │      │    MedQA_Examiner.examine()         │
       │      │   - executes each plan step         │
       │      │   - maintains short-term memory     │
       │      │   - analyzes options A/B/C/D        │
       │      └─────────────────────────────────────┘
       │                       │
       │                       ▼
       │      ┌─────────────────────────────────────┐
       │      │   agents/evaluator.py               │
       │      │   MedQA_Evaluator.evaluate()        │
       │      │   - verifies against guidelines     │
       │      │   - emits Continue/Complete/...     │
       │      └─────────────────────────────────────┘
       │                       │
       └───────────────────────┴───────────────────────
                                  ▼
                  ┌─────────────────────────┐
                  │   SolveResult (output)  │
                  │   predicted_answer      │
                  │   confidence            │
                  │   metadata (variant...) │
                  └─────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────┐
                  │ evaluation/runner.py    │
                  │ MedQAEvaluator.evaluate │
                  │   - runs all 5 variants │
                  │   - on 1273 questions   │
                  │   - extracts ≥20 errors │
                  │   - reports Δ(V3-V0)    │
                  └─────────────────────────┘
```

---

## 🧩 Module Responsibilities

### `core/system.py` — Orchestration

| Component | Responsibility |
|-----------|----------------|
| `MedQASystem` | Top-level entry point. Lazy-loads RAG + agents. Routes each `solve()` call to the requested variant. |
| `Variant` (enum) | `V0`, `V1`, `V2`, `V3`, `V4` |
| `SolveResult` | Dataclass holding prediction + correctness + metadata per question. |
| `_get_guidelines()` | Picks between standard RAG and Two-step Retrieval based on flag. |

Constructor flags:
```python
MedQASystem(
    api_key="sk-...",
    api_base="https://codex.dichvumang86.net/v1",  # optional
    use_two_step_retrieval=True,                  # optional
    valid_book_names=["Cardiology", "Ethics", ...], # optional
    keyword_model="gpt-4o-mini",                   # optional
)
```

### `agents/` — Multi-agent pipeline

| File | Class | Role |
|------|-------|------|
| `planner.py` | `MedQA_Planner` | Generates JSON reasoning plan: `recall → analysis → comparison → elimination → synthesis → final_answer`. |
| `examiner.py` | `MedQA_Examiner` | Executes the plan step-by-step. Maintains **short-term memory** of intermediate findings. Eliminates wrong options. |
| `evaluator.py` | `MedQA_Evaluator` | Verifies examiner's reasoning against medical guidelines. Returns `Continue / Revise / Complete / Terminate`. |

### `rag/retriever.py` — Retrieval

Two modes:

1. **Standard RAG** (`get_relevant_context`)
   - Direct semantic search over ChromaDB using question + options.
2. **Two-step Retrieval** (`get_relevant_context_two_step`) — *MedAgent-Pro style*
   - **Step 1**: LLM (`gpt-4o-mini` by default) extracts 2-3 medical keywords → filters out distractors like *"surgery"*, *"resident"*.
   - **Step 2**: Keywords used as vector query → ChromaDB's pre-injected `book_name` metadata guides L2 distance toward the right textbook domain.

### `rag/data_loader.py` — Dataset utilities

| Class | Role |
|-------|------|
| `MedQAQuestion` | Dataclass for one MedQA question. |
| `MedQALoader` | Loads from official JSON format. |
| `MedQAExporter` | JSON / CSV export. |
| `MedQABatchProcessor` | Batched processing helpers. |

### `evaluation/runner.py` — Benchmark

| Function / Class | Role |
|------------------|------|
| `MedQAEvaluator` | Runs all 5 variants on the same question set. |
| `run_evaluation()` | Convenience wrapper used by CLI / `evaluate` script. |
| `EvaluationMetrics` | Per-variant: accuracy, invalid rate, avg confidence. |
| `EvaluationReport` | Full report including `accuracy_gain = V3 - V0` and ≥20 error cases. |

CLI:
```bash
python -m medqa_rag.evaluation.runner \
    --data ./medqa_data/test.json \
    --output ./results \
    --variants V0 V1 V2 V3 V4
```

---

## 🎛️ 5-Variant Ablation Matrix

| Variant | RAG | Planner | Examiner | Memory | Evaluator |
|---------|:---:|:-------:|:--------:|:------:|:---------:|
| **V0** — Direct LLM           | ❌ | ❌ | ❌ | ❌ | ❌ |
| **V1** — RAG + Direct LLM     | ✅ | ❌ | ❌ | ❌ | ❌ |
| **V2** — Multi-agent no mem.  | ✅ | ✅ | ✅ | ❌ | ✅ |
| **V3** — Full system          | ✅ | ✅ | ✅ | ✅ | ✅ |
| **V4** — V3 without verifier  | ✅ | ✅ | ✅ | ✅ | ❌ |

Two-step Retrieval can be toggled on top of V1/V2/V3/V4:
```python
system = MedQASystem(api_key=..., use_two_step_retrieval=True)
result = system.solve(..., variant="V3")
```

---

## 🔐 Configuration

`config.py` reads `.env` automatically via `python-dotenv`.

### Default paths on this machine

| Resource                            | Path                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| ChromaDB (with metadata injection)  | `/Users/mac/Developers/MedQA_RAG/MedQA_ChromaDB_Injected`                             |
| MedQA-USMLE test set (JSONL)        | `/Users/mac/Developers/MedQA_RAG/dataset_MedQA-USMLE/questions/US/test.jsonl`         |

```env
# Required
OPENAI_API_KEY=sk-...

# Optional — third-party / proxy endpoint
OPENAI_API_BASE=https://codex.dichvumang86.net/v1

# Optional — model / RAG knobs
DEFAULT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
RAG_PERSIST_DIR=/Users/mac/Developers/MedQA_RAG/MedQA_ChromaDB_Injected
RAG_TOP_K=5
USE_HUGGINGFACE=false

# Optional — evaluation
MEDQA_TEST_PATH=/Users/mac/Developers/MedQA_RAG/dataset_MedQA-USMLE/questions/US/test.jsonl
MAX_QUESTIONS=1273
EVALUATION_OUTPUT_DIR=./results
```

---

## 📦 Public API Quick Reference

```python
from medqa_rag import (
    # System
    MedQASystem, SolveResult, Variant,
    # Agents
    MedQA_Planner, MedQA_Examiner, MedQA_Evaluator,
    ReasoningStep, ReasoningResult, OptionAnalysis,
    EvaluationStatus, VerificationResult,
    # RAG + Data
    MedQA_RAG, MedQALoader, MedQAQuestion,
    MedQAExporter, MedQABatchProcessor,
    # Evaluation
    MedQAEvaluator, run_evaluation,
    EvaluationMetrics, EvaluationReport,
    # Config
    load_config, get_api_key, require_api_key,
    Config, get_rag_config, get_model_config,
)
```

Or, from sub-packages:
```python
from medqa_rag.agents import MedQA_Planner
from medqa_rag.rag import MedQA_RAG
from medqa_rag.core import MedQASystem
from medqa_rag.evaluation import run_evaluation
```

---

## 🚀 Typical Usage

### Single question
```python
from medqa_rag import MedQASystem, load_config

load_config()
system = MedQASystem(use_two_step_retrieval=True)

result = system.solve(
    question="A 65-year-old man with heart failure...",
    options={"A": "...", "B": "...", "C": "...", "D": "..."},
    correct_answer="B",
    question_id="q001",
    variant="V3",
)

print(result.predicted_answer, result.is_correct, result.confidence)
```

### Full benchmark
```python
from medqa_rag import run_evaluation

report = run_evaluation(
    data_path="./medqa_data/test.json",
    output_dir="./results",
    max_questions=1273,
)
print(report.summary())
```
