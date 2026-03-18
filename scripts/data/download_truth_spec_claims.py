#!/usr/bin/env python3
"""
Download truth_spec claims CSVs into data/apollo_raw.

The upstream repository stores the 4 claims datasets as CSV files under:
  https://github.com/zfying/truth_spec/tree/main/data/claims
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/zfying/truth_spec/main/data/claims"

DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "claims_definitional": {
        "dataset_name": "Deception-ClaimsDefinitional",
        "filename": "claims__definitional_gemini_600_full.csv",
    },
    "claims_evidential": {
        "dataset_name": "Deception-ClaimsEvidential",
        "filename": "claims__evidential_gemini_600_full.csv",
    },
    "claims_fictional": {
        "dataset_name": "Deception-ClaimsFictional",
        "filename": "claims__fictional_gemini_600_full.csv",
    },
    "claims_logical": {
        "dataset_name": "Deception-ClaimsLogical",
        "filename": "claims__logical_gemini_600_full.csv",
    },
}


def download_file(url: str, output_path: Path, overwrite: bool) -> Dict[str, str]:
    if output_path.exists() and not overwrite:
        logger.info("skip existing %s", output_path)
        return {"status": "skipped", "path": str(output_path), "url": url}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    logger.info("saved %s (%.1f KB)", output_path, len(response.content) / 1024.0)
    return {"status": "downloaded", "path": str(output_path), "url": url}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download truth_spec claims CSVs.")
    parser.add_argument("--output_dir", type=str, default="data/apollo_raw")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_SPECS.keys()),
        help=f"Comma-separated subset of: {','.join(DATASET_SPECS.keys())}",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in requested if x not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    manifest = {
        "output_root": str(output_root),
        "overwrite": bool(args.overwrite),
        "datasets": [],
    }
    for slug in requested:
        spec = DATASET_SPECS[slug]
        filename = spec["filename"]
        url = f"{RAW_BASE}/{filename}"
        output_path = output_root / slug / filename
        result = download_file(url, output_path, overwrite=args.overwrite)
        manifest["datasets"].append(
            {
                "slug": slug,
                "dataset_name": spec["dataset_name"],
                "filename": filename,
                **result,
            }
        )

    manifest_path = output_root / "truth_spec_claims_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote manifest -> %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
