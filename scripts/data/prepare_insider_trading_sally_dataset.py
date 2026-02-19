#!/usr/bin/env python3
"""
Prepare Insider Trading FINAL JSONL into Sally-only concatenated examples.

Target format (aligned with existing deception datasets):
    {
        "prompt": "<system_prompt>\\n\\n<sally_user_messages_concat>",
        "completion": "<final send_message report body>",
        "gold_label": 0|1,   # 0=non-deceptive, 1=deceptive
        "split": "train|validation|test",
        "metadata": {...}
    }

Inputs expected per row:
    - inputs: list[{role, content}, ...]
    - labels: "positive" | "negative"
    - scale_labels, scale_label_explanation, ...

This script writes:
    - sally_concat_records.jsonl   (extracted intermediate fields)
    - sally_concat_examples.jsonl  (prompt/completion/gold_label)
    - sally_concat_stats.json      (summary metadata)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional


SALLY_EMAIL_RE = re.compile(r"from:\s*sally@whitestone\.com", re.IGNORECASE)
MESSAGE_BLOCK_RE = re.compile(r'Message:\s*"""(.*?)"""', re.DOTALL)

# Common send_message action input patterns observed in this dataset.
SEND_MESSAGE_BODY_PATTERNS = [
    re.compile(r'Action Input:\s*"[^"]*"\s*,\s*"""(.*?)"""', re.DOTALL),
    re.compile(r"Action Input:\s*[^,\n]+,\s*\"\"\"(.*?)\"\"\"", re.DOTALL),
]


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_system_prompt(inputs: List[Dict]) -> str:
    for msg in inputs:
        if msg.get("role") == "system":
            return str(msg.get("content", "")).strip()
    return ""


def extract_sally_messages_concat(inputs: List[Dict]) -> str:
    sally_chunks: List[str] = []
    for msg in inputs:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        blocks = MESSAGE_BLOCK_RE.findall(content)

        if blocks:
            for block in blocks:
                block = block.strip()
                if block and SALLY_EMAIL_RE.search(block):
                    sally_chunks.append(f'Message: """{block}"""')
            continue

        if SALLY_EMAIL_RE.search(content):
            sally_chunks.append(content.strip())

    return "\n\n".join(chunk for chunk in sally_chunks if chunk).strip()


def extract_last_assistant_completion(inputs: List[Dict]) -> str:
    for msg in reversed(inputs):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def extract_report_response(inputs: List[Dict]) -> str:
    for msg in reversed(inputs):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        if "Action: send_message" not in content:
            continue

        for pattern in SEND_MESSAGE_BODY_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                return matches[-1].strip()

        # Fallback to raw assistant message if we cannot parse body cleanly.
        return content.strip()

    return ""


def map_gold_label(row: Dict) -> Optional[int]:
    raw = str(row.get("labels", "")).strip().lower()
    if raw == "negative":
        return 1  # deceptive
    if raw == "positive":
        return 0  # non-deceptive

    scale = row.get("scale_labels")
    try:
        scale_num = float(scale)
    except (TypeError, ValueError):
        return None
    return 1 if scale_num >= 5 else 0


def stratified_split_indices(labels: List[int], seed: int) -> Dict[int, str]:
    by_label: Dict[int, List[int]] = {0: [], 1: []}
    for idx, lbl in enumerate(labels):
        if lbl in by_label:
            by_label[lbl].append(idx)

    rng = random.Random(seed)
    split_map: Dict[int, str] = {}
    for lbl, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)
        for i, idx in enumerate(idxs):
            if i < train_end:
                split_map[idx] = "train"
            elif i < val_end:
                split_map[idx] = "validation"
            else:
                split_map[idx] = "test"
    return split_map


def build_records(
    raw_rows: List[Dict],
    max_rows: Optional[int] = None,
    split_seed: int = 42,
) -> Dict[str, object]:
    records: List[Dict] = []
    skipped = {
        "missing_inputs": 0,
        "missing_system_prompt": 0,
        "missing_sally_messages": 0,
        "missing_report_response": 0,
        "missing_label": 0,
    }

    selected_rows = raw_rows if max_rows is None else raw_rows[:max_rows]

    for idx, row in enumerate(selected_rows):
        inputs = row.get("inputs")
        if not isinstance(inputs, list):
            skipped["missing_inputs"] += 1
            continue

        system_prompt = extract_system_prompt(inputs)
        if not system_prompt:
            skipped["missing_system_prompt"] += 1
            continue

        sally_concat = extract_sally_messages_concat(inputs)
        if not sally_concat:
            skipped["missing_sally_messages"] += 1
            continue

        report_response = extract_report_response(inputs)
        if not report_response:
            skipped["missing_report_response"] += 1
            continue

        gold_label = map_gold_label(row)
        if gold_label is None:
            skipped["missing_label"] += 1
            continue

        labels_raw = str(row.get("labels", "")).strip().lower()
        report_label = "deceptive" if labels_raw == "negative" else "non_deceptive"
        if labels_raw not in {"negative", "positive"}:
            report_label = "deceptive" if gold_label == 1 else "non_deceptive"

        records.append(
            {
                "id": row.get("ids", idx),
                "system_prompt": system_prompt,
                "user_prompt_sally_concat": sally_concat,
                "report_response": report_response,
                "assistant_last_completion": extract_last_assistant_completion(inputs),
                "gold_label": gold_label,
                "report_label": report_label,
                "labels_raw": row.get("labels"),
                "scale_labels_raw": row.get("scale_labels"),
                "scale_label_explanation": row.get("scale_label_explanation"),
                "scale_label_confidence": row.get("scale_label_confidence"),
                "scale_label_model": row.get("scale_label_model"),
                "n_messages": len(inputs),
            }
        )

    labels = [int(r["gold_label"]) for r in records]
    split_map = stratified_split_indices(labels, seed=split_seed)
    for i, rec in enumerate(records):
        rec["split"] = split_map.get(i, "train")

    return {
        "records": records,
        "skipped": skipped,
        "total_rows": len(selected_rows),
    }


def build_examples(records: List[Dict]) -> List[Dict]:
    examples: List[Dict] = []
    for rec in records:
        system_prompt = rec["system_prompt"].strip()
        user_prompt = rec["user_prompt_sally_concat"].strip()
        prompt_parts = [system_prompt, user_prompt]
        prompt = "\n\n".join(p for p in prompt_parts if p)
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        examples.append(
            {
                "prompt": prompt,
                "completion": rec["report_response"],
                "gold_label": int(rec["gold_label"]),
                "split": rec["split"],
                "metadata": {
                    "dataset": "Deception-InsiderTrading-SallyConcat",
                    "id": f"insider_sally_{rec['id']}",
                    "source_id": rec["id"],
                    "report_label": rec["report_label"],
                    "labels_raw": rec["labels_raw"],
                    "scale_labels_raw": rec["scale_labels_raw"],
                    "scale_label_explanation": rec["scale_label_explanation"],
                    "scale_label_confidence": rec["scale_label_confidence"],
                    "scale_label_model": rec["scale_label_model"],
                    "n_messages": rec["n_messages"],
                    "input_strategy": "system_plus_sally_concat",
                    "input_messages": input_messages,
                    "has_pregenerated_response": True,
                },
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Sally-only Insider Trading dataset from FINAL JSONL"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to Insider_trading_final.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/apollo_raw/insider_trading",
        help="Output directory for prepared artifacts",
    )
    parser.add_argument(
        "--records_name",
        type=str,
        default="sally_concat_records.jsonl",
        help="Filename for extracted intermediate records",
    )
    parser.add_argument(
        "--examples_name",
        type=str,
        default="sally_concat_examples.jsonl",
        help="Filename for prompt/completion examples",
    )
    parser.add_argument(
        "--stats_name",
        type=str,
        default="sally_concat_stats.json",
        help="Filename for summary stats",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on number of input rows to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/validation/test split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist",
    )
    args = parser.parse_args()

    records_path = os.path.join(args.output_dir, args.records_name)
    examples_path = os.path.join(args.output_dir, args.examples_name)
    stats_path = os.path.join(args.output_dir, args.stats_name)

    existing = [p for p in [records_path, examples_path, stats_path] if os.path.exists(p)]
    if existing and not args.overwrite:
        print("Output artifacts already exist. Re-run with --overwrite to regenerate.")
        for path in existing:
            print(f"  - {path}")
        return

    raw_rows = read_jsonl(args.input_jsonl)
    payload = build_records(raw_rows, max_rows=args.max_rows, split_seed=args.seed)
    records = payload["records"]
    skipped = payload["skipped"]
    total_rows = payload["total_rows"]

    examples = build_examples(records)

    write_jsonl(records_path, records)
    write_jsonl(examples_path, examples)

    n_non_deceptive = sum(1 for ex in examples if ex["gold_label"] == 0)
    n_deceptive = sum(1 for ex in examples if ex["gold_label"] == 1)

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": args.input_jsonl,
        "output_dir": args.output_dir,
        "records_path": records_path,
        "examples_path": examples_path,
        "total_input_rows_considered": total_rows,
        "total_records_written": len(records),
        "total_examples_written": len(examples),
        "split_seed": args.seed,
        "label_counts": {
            "non_deceptive_0": n_non_deceptive,
            "deceptive_1": n_deceptive,
        },
        "split_counts": {
            "train": sum(1 for ex in examples if ex["split"] == "train"),
            "validation": sum(1 for ex in examples if ex["split"] == "validation"),
            "test": sum(1 for ex in examples if ex["split"] == "test"),
        },
        "skipped": skipped,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Prepared Sally-only Insider Trading dataset")
    print(f"  Records:  {records_path}")
    print(f"  Examples: {examples_path}")
    print(f"  Stats:    {stats_path}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Labels -> non-deceptive: {n_non_deceptive}, deceptive: {n_deceptive}")


if __name__ == "__main__":
    main()
