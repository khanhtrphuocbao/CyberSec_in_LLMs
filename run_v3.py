#!/usr/bin/env python3
"""CLI entry point for the resumable, parallel V3 MedQA benchmark."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional


def _load_local_package() -> None:
    """Load this checkout as ``medqa_rag`` regardless of its directory name."""
    if "medqa_rag" in sys.modules:
        return
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "medqa_rag", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the local medqa_rag package")
    package = importlib.util.module_from_spec(spec)
    sys.modules["medqa_rag"] = package
    spec.loader.exec_module(package)


_load_local_package()

from medqa_rag.evaluation.v2_benchmark import V3BenchmarkRunner, build_v2_parser
from medqa_rag.rag.data_loader import MedQALoader
from medqa_rag.core.system import MedQASystem
from medqa_rag.config import load_config


def build_v3_parser():
    """Build the V3 CLI with its worker and output defaults."""
    parser = build_v2_parser()
    parser.description = "Run the resumable V3 MedQA benchmark"
    parser.set_defaults(output_dir="results_V3", cache_file="results_V3/rag_cache.json", workers=2)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key to dedicate to V3",
    )
    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "Directory for results_V3.json"
        elif action.dest == "workers":
            action.help = "Concurrent question workers (default: 2)"
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_v3_parser()
    args = parser.parse_args(argv)
    load_config()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Environment variable {args.api_key_env} is not set")
    book_names = args.valid_book_names.split(",") if args.valid_book_names else None
    questions = MedQALoader().load_json(args.data_path)
    runner = V3BenchmarkRunner(
        questions,
        Path(args.output_dir),
        workers=args.workers,
        cache_path=Path(args.cache_file),
        top_k=args.top_k,
        use_two_step_retrieval=args.two_step_retrieval,
        valid_book_names=book_names,
        system_factory=lambda: MedQASystem(api_key=api_key),
    )
    results = runner.run()
    valid_results = [result for result in results if result.get("is_valid") is not False]
    correct = sum(1 for result in valid_results if result.get("is_correct"))
    accuracy = correct / len(valid_results) * 100 if valid_results else 0.0
    print(f"DONE! {len(valid_results)}/{len(questions)} | {correct} correct | {accuracy:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
