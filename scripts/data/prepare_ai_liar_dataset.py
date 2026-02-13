#!/usr/bin/env python3
"""
Prepare paired AILiar dataset from Apollo rollouts.

Creates two outputs:
1) paired_rollouts.jsonl: one row per rollout with one honest + one deceptive output
2) paired_examples.jsonl: two rows per rollout (honest + deceptive), ready for caching

Pairing rule: first honest + first deceptive (deterministic).
Splits are rollout-level: 60/20/20 train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def first_index(labels: List[str], target: str) -> Optional[int]:
    for i, lbl in enumerate(labels):
        if str(lbl).strip().lower() == target:
            return i
    return None


def extract_texts(rollout: dict) -> List[str]:
    if "output_strs" in rollout and isinstance(rollout["output_strs"], list):
        return rollout["output_strs"]
    out_msgs = rollout.get("output_messages", [])
    return [m.get("content", "") for m in out_msgs]


def build_prompt_from_input_messages(input_messages: List[dict]) -> str:
    # Simple fallback prompt string (system + user). Tokenization for caching uses chat template.
    system = ""
    users = []
    for msg in input_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system" and not system:
            system = content
        elif role == "user":
            users.append(content)
    parts = []
    if system:
        parts.append(system.strip())
    if users:
        parts.append("\n\n".join(u.strip() for u in users if u.strip()))
    return "\n\n".join(p for p in parts if p)


def pair_rollouts(raw: dict, max_pairs: int, seed: int) -> Tuple[List[dict], List[dict]]:
    rollouts = raw.get("rollouts", [])
    paired = []

    for idx, rollout in enumerate(rollouts):
        deception_pct = rollout.get("deception_percentage")
        if deception_pct is None:
            continue
        if not (0 < float(deception_pct) <= 0.8):
            continue

        graded = rollout.get("graded_deception", [])
        if not graded:
            continue

        honest_idx = first_index(graded, "honest")
        deceptive_idx = first_index(graded, "deceptive")
        if honest_idx is None or deceptive_idx is None:
            continue

        texts = extract_texts(rollout)
        if honest_idx >= len(texts) or deceptive_idx >= len(texts):
            continue

        input_messages = rollout.get("input_messages", [])
        system_msg = next((m.get("content", "") for m in input_messages if m.get("role") == "system"), "")
        user_msgs = [m.get("content", "") for m in input_messages if m.get("role") == "user"]

        paired.append(
            {
                "rollout_id": f"ai_liar_{idx}",
                "system": system_msg,
                "users": user_msgs,
                "input_messages": input_messages,
                "honest_index": honest_idx,
                "deceptive_index": deceptive_idx,
                "honest_output": texts[honest_idx],
                "deceptive_output": texts[deceptive_idx],
                "deception_percentage": deception_pct,
                "graded_deception": graded,
            }
        )

    random.seed(seed)
    random.shuffle(paired)
    paired = paired[:max_pairs]

    # Rollout-level split
    n = len(paired)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    for i, row in enumerate(paired):
        if i < train_end:
            row["split"] = "train"
        elif i < val_end:
            row["split"] = "validation"
        else:
            row["split"] = "test"

    examples = []
    for row in paired:
        prompt = build_prompt_from_input_messages(row["input_messages"])
        rid = row["rollout_id"]
        split = row["split"]
        base_meta = {
            "dataset": "Deception-AILiar",
            "rollout_id": rid,
            "split": split,
            "input_messages": row["input_messages"],
            "graded_deception": row["graded_deception"],
            "deception_percentage": row["deception_percentage"],
            "has_pregenerated_response": True,
        }

        examples.append(
            {
                "prompt": prompt,
                "completion": row["honest_output"],
                "gold_label": 0,
                "metadata": {**base_meta, "id": f"{rid}_honest", "output_index": row["honest_index"]},
                "split": split,
            }
        )
        examples.append(
            {
                "prompt": prompt,
                "completion": row["deceptive_output"],
                "gold_label": 1,
                "metadata": {**base_meta, "id": f"{rid}_deceptive", "output_index": row["deceptive_index"]},
                "split": split,
            }
        )

    return paired, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired AILiar dataset from Apollo rollouts")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to ai_liar__original_without_answers__llama-70b.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write paired_rollouts.jsonl and paired_examples.jsonl",
    )
    parser.add_argument("--max_pairs", type=int, default=200, help="Number of paired rollouts to keep")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for splits")
    args = parser.parse_args()

    raw = read_json(args.input_json)
    paired, examples = pair_rollouts(raw, max_pairs=int(args.max_pairs), seed=int(args.seed))

    paired_path = os.path.join(args.output_dir, "paired_rollouts.jsonl")
    examples_path = os.path.join(args.output_dir, "paired_examples.jsonl")

    write_jsonl(paired_path, paired)
    write_jsonl(examples_path, examples)

    print(f"Wrote {len(paired)} paired rollouts -> {paired_path}")
    print(f"Wrote {len(examples)} examples -> {examples_path}")


if __name__ == "__main__":
    main()
