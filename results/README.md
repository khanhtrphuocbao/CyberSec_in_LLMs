# Benchmark results

Canonical benchmark artifacts are organized by variant:

- `V0/` — direct LLM baseline
- `V1/` — RAG-only
- `V2/` — multi-agent without memory
- `V3/` — full multi-agent system
- `V4/` — full system without evaluator

Legacy paths (`results_V0`, `results_V1`, `results_V2`, and
`medqa_rag/results_V3`, `medqa_rag/results_V4`) are compatibility symlinks to
these directories, so existing benchmark runners can resume without changing
their output configuration.
