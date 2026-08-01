# Single-question CLI design

## Goal

Provide reproducible command-line entry points for V0 through V4 that run one
MedQA test-set row by zero-based index, using `OPENAI_API_KEY` by default.

## Design

`single_question_cli.py` will own the common parsing and execution flow. It
loads the selected row, constructs one `MedQASystem`, calls `solve()` with the
requested variant, and stores the serialised `SolveResult` without using the
resumable benchmark runners.

The existing V2, V3, and V4 wrappers keep their current full-benchmark mode
when `--question-index` is absent. When it is present, they delegate to the
common single-question flow. New V0 and V1 wrappers delegate to that same
flow.

## Output and safety

The selected question is written to
`results/single_question/<variant>/<question_id>.json`; stdout and stderr from
the solve are mirrored to `logs/single_question/<variant>/<question_id>.log`.
These paths never overlap the resumable `results_V*/results_V*.json` files.
An invalid index produces a parser error before an API request.

## Verification

Unit tests use a fake loader and fake system to verify variant routing,
zero-based index selection, persisted JSON/log output, and index validation.
The live smoke test runs index 10 separately for V0, V1, V2, V3, and V4 after
the offline suite passes.
