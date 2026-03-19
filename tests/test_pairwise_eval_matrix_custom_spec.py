from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pipelines" / "run_pairwise_eval_matrix.py"

spec = importlib.util.spec_from_file_location("pairwise_eval_matrix", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_stage_spec_supports_custom_completion_datasets() -> None:
    args = argparse.Namespace(
        completion_rows="Deception-Roleplaying-completion,Deception-InsiderTrading-SallyConcat-completion",
        completion_cols="Deception-Roleplaying-completion,Deception-InsiderTrading-SallyConcat-completion",
        full_rows=None,
        full_cols=None,
    )

    rows, cols = module.stage_spec(args)

    assert rows["completion"] == [
        "Deception-Roleplaying-completion",
        "Deception-InsiderTrading-SallyConcat-completion",
    ]
    assert cols["completion"] == [
        "Deception-Roleplaying-completion",
        "Deception-InsiderTrading-SallyConcat-completion",
    ]
    assert "Deception-InsiderTrading-full" in cols["full"]


def test_short_name_maps_sally_insider_trading() -> None:
    assert module.short_name("Deception-InsiderTrading-SallyConcat-completion") == "ITS-c"
