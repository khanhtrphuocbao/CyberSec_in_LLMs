#!/usr/bin/env python3
"""CLI entry point for the resumable, parallel V2 MedQA benchmark."""

import importlib.util
import sys
from pathlib import Path


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

from medqa_rag.evaluation.v2_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
