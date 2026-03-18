from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "actprobe" / "src"))

from actprobe.datasets.deception_loaders import (  # noqa: E402
    DeceptionClaimsEvidentialDataset,
)


PREP_SCRIPT_PATH = REPO_ROOT / "scripts" / "data" / "prepare_truth_spec_claims.py"
prep_spec = importlib.util.spec_from_file_location("prepare_truth_spec_claims", PREP_SCRIPT_PATH)
assert prep_spec is not None and prep_spec.loader is not None
prep_module = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prep_module)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_normalize_rows_creates_two_examples_per_claim() -> None:
    rows = [
        {
            "Claim": "Paris is the capital of France.",
            "Negated Claim": "Paris is not the capital of France.",
            "Domain": "Geography",
        }
    ]
    spec = prep_module.DATASET_SPECS["claims_evidential"]
    normalized = prep_module.normalize_rows("claims_evidential", spec, rows)

    assert len(normalized) == 2
    assert normalized[0]["label"] == 0
    assert normalized[1]["label"] == 1
    assert normalized[0]["messages_clean"][0]["content"] == prep_module.PROMPT_TEXT
    assert normalized[0]["messages_clean"][1]["content"] == "Paris is the capital of France."
    assert normalized[1]["messages_clean"][1]["content"] == "Paris is not the capital of France."
    assert normalized[0]["domain"] == "Geography"


def test_claims_loader_accepts_matching_rows(tmp_path: Path) -> None:
    path = tmp_path / "claims_evidential__truth_spec__messages_clean.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "claims_evidential_0_claim",
                "dataset": "Deception-ClaimsEvidential",
                "label": 0,
                "messages_clean": [
                    {"role": "user", "content": "Please tell me a fact."},
                    {"role": "assistant", "content": "Paris is the capital of France."},
                ],
                "domain": "Geography",
            },
            {
                "id": "claims_evidential_0_negated_claim",
                "dataset": "Deception-ClaimsEvidential",
                "label": 1,
                "messages_clean": [
                    {"role": "user", "content": "Please tell me a fact."},
                    {"role": "assistant", "content": "Paris is not the capital of France."},
                ],
                "domain": "Geography",
            },
        ],
    )

    ds = DeceptionClaimsEvidentialDataset(split="train", data_file=str(path))
    ds.load_data()

    assert len(ds) == 2
    assert ds[0]["metadata"]["dataset"] == "Deception-ClaimsEvidential"
    assert ds[0]["metadata"]["id"].startswith("claims_evidential_")


def test_claims_loader_rejects_wrong_claims_dataset_rows(tmp_path: Path) -> None:
    path = tmp_path / "claims_evidential__truth_spec__messages_clean.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "claims_logical_0_claim",
                "dataset": "Deception-ClaimsLogical",
                "label": 0,
                "messages_clean": [
                    {"role": "user", "content": "Please tell me a fact."},
                    {"role": "assistant", "content": "2 + 2 = 4"},
                ],
            }
        ],
    )

    ds = DeceptionClaimsEvidentialDataset(split="train", data_file=str(path))
    with pytest.raises(ValueError, match="inconsistent with Deception-ClaimsEvidential"):
        ds.load_data()
