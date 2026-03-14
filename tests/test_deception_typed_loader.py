from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "actprobe" / "src"))

from actprobe.datasets.deception_loaders import (  # noqa: E402
    DeceptionInstructedDeceptionDataset,
    DeceptionMaskDataset,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sample_mask_row() -> dict:
    return {
        "id": "mask_1000",
        "dataset": "Deception-Mask",
        "split": "train",
        "label": 1,
        "messages_clean": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ],
    }


def test_instructed_loader_does_not_fallback_to_mask_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_exists(path: str) -> bool:
        return "mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl" in path

    monkeypatch.setattr("actprobe.datasets.deception_loaders.os.path.exists", fake_exists)

    with pytest.raises(FileNotFoundError):
        DeceptionInstructedDeceptionDataset(split="train")


def test_instructed_loader_rejects_mask_rows(tmp_path: Path) -> None:
    path = tmp_path / "typed.jsonl"
    write_jsonl(path, [sample_mask_row()])

    ds = DeceptionInstructedDeceptionDataset(split="train", data_file=str(path))
    with pytest.raises(ValueError, match="inconsistent with Deception-InstructedDeception"):
        ds.load_data()


def test_mask_loader_accepts_mask_rows(tmp_path: Path) -> None:
    path = tmp_path / "mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl"
    write_jsonl(path, [sample_mask_row()])

    ds = DeceptionMaskDataset(split="train", data_file=str(path))
    ds.load_data()

    assert len(ds) == 1
    assert ds[0]["metadata"]["id"].startswith("mask_")
    assert ds[0]["gold_label"] == 1
