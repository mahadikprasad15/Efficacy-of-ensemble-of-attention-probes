#!/usr/bin/env python3
"""
Inspect and normalize truth_spec claims CSVs into typed-message JSONL files.

Each CSV row becomes two examples:
  - Claim -> honest label 0
  - Negated Claim -> deceptive label 1
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROMPT_TEXT = "Please tell me a fact."

DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "claims_definitional": {
        "dataset_name": "Deception-ClaimsDefinitional",
        "filename": "claims__definitional_gemini_600_full.csv",
        "normalized_filename": "claims_definitional__truth_spec__messages_clean.jsonl",
        "id_prefix": "claims_definitional",
    },
    "claims_evidential": {
        "dataset_name": "Deception-ClaimsEvidential",
        "filename": "claims__evidential_gemini_600_full.csv",
        "normalized_filename": "claims_evidential__truth_spec__messages_clean.jsonl",
        "id_prefix": "claims_evidential",
    },
    "claims_fictional": {
        "dataset_name": "Deception-ClaimsFictional",
        "filename": "claims__fictional_gemini_600_full.csv",
        "normalized_filename": "claims_fictional__truth_spec__messages_clean.jsonl",
        "id_prefix": "claims_fictional",
    },
    "claims_logical": {
        "dataset_name": "Deception-ClaimsLogical",
        "filename": "claims__logical_gemini_600_full.csv",
        "normalized_filename": "claims_logical__truth_spec__messages_clean.jsonl",
        "id_prefix": "claims_logical",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and normalize truth_spec claims CSVs.")
    parser.add_argument("--input_root", type=str, default="data/apollo_raw")
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_SPECS.keys()),
        help=f"Comma-separated subset of: {','.join(DATASET_SPECS.keys())}",
    )
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def inspection_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    null_counts: Dict[str, int] = {c: 0 for c in columns}
    for row in rows:
        for key in columns:
            value = row.get(key)
            if value is None or str(value).strip() == "":
                null_counts[key] += 1
    return {
        "row_count": len(rows),
        "columns": columns,
        "null_counts": null_counts,
        "sample_rows": rows[:3],
    }


def normalize_rows(slug: str, spec: Dict[str, str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset_name = spec["dataset_name"]
    id_prefix = spec["id_prefix"]
    normalized: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        claim = (row.get("Claim") or "").strip()
        negated = (row.get("Negated Claim") or "").strip()
        domain = (row.get("Domain") or "").strip()
        if not claim or not negated:
            continue

        base_meta = {
            "dataset": dataset_name,
            "source_repo": "truth_spec",
            "source_slug": slug,
            "row_index": idx,
            "domain": domain,
        }

        normalized.append(
            {
                "id": f"{id_prefix}_{idx}_claim",
                "dataset": dataset_name,
                "label": 0,
                "messages_clean": [
                    {"role": "user", "content": PROMPT_TEXT},
                    {"role": "assistant", "content": claim},
                ],
                "polarity": "claim",
                **base_meta,
            }
        )
        normalized.append(
            {
                "id": f"{id_prefix}_{idx}_negated_claim",
                "dataset": dataset_name,
                "label": 1,
                "messages_clean": [
                    {"role": "user", "content": PROMPT_TEXT},
                    {"role": "assistant", "content": negated},
                ],
                "polarity": "negated_claim",
                **base_meta,
            }
        )

    return normalized


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    run_id = args.run_id.strip() or f"{default_run_id()}-truth-spec-claims-prep"
    run_root = (
        Path(args.artifact_root)
        / "runs"
        / "prepare_truth_spec_claims"
        / "truth-spec-claims"
        / "fixed-fact-prompt"
        / run_id
    )
    inputs_dir = run_root / "inputs"
    results_dir = run_root / "results"
    meta_dir = run_root / "meta"
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    for d in [inputs_dir, results_dir, meta_dir, checkpoints_dir, logs_dir]:
        ensure_dir(d)

    status_path = meta_dir / "status.json"
    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "input_root": str(input_root),
            "inspect_only": bool(args.inspect_only),
            "overwrite": bool(args.overwrite),
            "prompt_text": PROMPT_TEXT,
            "datasets": [x.strip() for x in args.datasets.split(",") if x.strip()],
            "run_root": str(run_root),
        },
    )
    write_json(status_path, {"state": "running", "message": "preparing truth_spec claims", "updated_at": utc_now()})

    requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in requested if x not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    progress = {"completed_datasets": []}
    write_json(checkpoints_dir / "progress.json", progress)

    inspection_rows: List[Dict[str, Any]] = []
    prepared_manifest: List[Dict[str, Any]] = []

    for slug in requested:
        spec = DATASET_SPECS[slug]
        raw_path = input_root / slug / spec["filename"]
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw CSV: {raw_path}")

        rows = load_csv_rows(raw_path)
        inspect_payload = inspection_summary(rows)
        inspect_payload.update(
            {
                "slug": slug,
                "dataset_name": spec["dataset_name"],
                "raw_csv_path": str(raw_path),
            }
        )
        inspection_rows.append(inspect_payload)
        write_json(results_dir / f"{slug}_inspection.json", inspect_payload)

        normalized_path = input_root / slug / spec["normalized_filename"]
        normalized_rows: List[Dict[str, Any]] = []
        if not args.inspect_only:
            normalized_rows = normalize_rows(slug, spec, rows)
            if normalized_path.exists() and not args.overwrite:
                pass
            else:
                write_jsonl(normalized_path, normalized_rows)

        prepared_manifest.append(
            {
                "slug": slug,
                "dataset_name": spec["dataset_name"],
                "raw_csv_path": str(raw_path),
                "normalized_jsonl_path": str(normalized_path),
                "normalized_row_count": len(normalized_rows) if normalized_rows else "",
                "label_0_count": sum(1 for row in normalized_rows if row.get("label") == 0) if normalized_rows else "",
                "label_1_count": sum(1 for row in normalized_rows if row.get("label") == 1) if normalized_rows else "",
            }
        )
        progress["completed_datasets"].append(slug)
        write_json(checkpoints_dir / "progress.json", progress)

    with (results_dir / "prepared_dataset_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slug",
                "dataset_name",
                "raw_csv_path",
                "normalized_jsonl_path",
                "normalized_row_count",
                "label_0_count",
                "label_1_count",
            ],
        )
        writer.writeheader()
        for row in prepared_manifest:
            writer.writerow(row)

    write_json(
        results_dir / "inspection_summary.json",
        {
            "run_id": run_id,
            "completed_at": utc_now(),
            "prompt_text": PROMPT_TEXT,
            "datasets": inspection_rows,
            "inspect_only": bool(args.inspect_only),
        },
    )
    write_json(status_path, {"state": "completed", "message": "completed successfully", "updated_at": utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
