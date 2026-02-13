#!/usr/bin/env python3
"""
Prepare AILiar dataset from Apollo rollouts.

Modes:
1) paired: one honest + one deceptive output per rollout (first of each).
2) all_labeled: include every output labeled honest/deceptive.

Creates two outputs:
- paired_rollouts.jsonl: one row per rollout (mode-dependent payload)
- paired_examples.jsonl: one row per example (prompt + completion + label)

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


def all_labeled_rollouts(raw: dict, max_rollouts: int, seed: int) -> Tuple[List[dict], List[dict]]:
    rollouts = raw.get("rollouts", [])
    grouped = []

    for idx, rollout in enumerate(rollouts):
        graded = rollout.get("graded_deception", [])
        if not graded:
            continue

        texts = extract_texts(rollout)
        if not texts:
            continue

        input_messages = rollout.get("input_messages", [])
        system_msg = next((m.get("content", "") for m in input_messages if m.get("role") == "system"), "")
        user_msgs = [m.get("content", "") for m in input_messages if m.get("role") == "user"]

        labeled_outputs = []
        for i, lbl in enumerate(graded):
            if i >= len(texts):
                break
            label = str(lbl).strip().lower()
            if label not in ("honest", "deceptive"):
                continue
            labeled_outputs.append(
                {
                    "output_index": i,
                    "label_str": label,
                    "output": texts[i],
                }
            )

        if not labeled_outputs:
            continue

        grouped.append(
            {
                "rollout_id": f"ai_liar_{idx}",
                "system": system_msg,
                "users": user_msgs,
                "input_messages": input_messages,
                "labeled_outputs": labeled_outputs,
                "deception_percentage": rollout.get("deception_percentage"),
                "graded_deception": graded,
            }
        )

    random.seed(seed)
    random.shuffle(grouped)
    grouped = grouped[:max_rollouts]

    # Rollout-level split
    n = len(grouped)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    for i, row in enumerate(grouped):
        if i < train_end:
            row["split"] = "train"
        elif i < val_end:
            row["split"] = "validation"
        else:
            row["split"] = "test"

    examples = []
    for row in grouped:
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

        for lo in row["labeled_outputs"]:
            gold = 0 if lo["label_str"] == "honest" else 1
            examples.append(
                {
                    "prompt": prompt,
                    "completion": lo["output"],
                    "gold_label": gold,
                    "metadata": {**base_meta, "id": f"{rid}_{lo['label_str']}_{lo['output_index']}", "output_index": lo["output_index"]},
                    "split": split,
                }
            )

    return grouped, examples


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
    parser.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "all_labeled"],
        help="Dataset mode: paired (one honest+deceptive per rollout) or all_labeled (all outputs with labels)",
    )
    parser.add_argument("--max_pairs", type=int, default=200, help="Number of paired rollouts to keep")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on total examples AFTER expansion (per split: 60/20/20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for splits")
    args = parser.parse_args()

    raw = read_json(args.input_json)
    if args.mode == "paired":
        paired, examples = pair_rollouts(raw, max_pairs=int(args.max_pairs), seed=int(args.seed))
    else:
        paired, examples = all_labeled_rollouts(raw, max_rollouts=int(args.max_pairs), seed=int(args.seed))

    if args.max_examples is not None:
        random.seed(args.seed)
        train_cap = int(0.6 * args.max_examples)
        val_cap = int(0.2 * args.max_examples)
        test_cap = args.max_examples - train_cap - val_cap
        by_split = {"train": [], "validation": [], "test": []}
        for ex in examples:
            by_split[ex.get("split", "train")].append(ex)
        for split, cap in [("train", train_cap), ("validation", val_cap), ("test", test_cap)]:
            random.shuffle(by_split[split])
            by_split[split] = by_split[split][:cap]
        examples = by_split["train"] + by_split["validation"] + by_split["test"]

    paired_path = os.path.join(args.output_dir, "paired_rollouts.jsonl")
    examples_path = os.path.join(args.output_dir, "paired_examples.jsonl")

    write_jsonl(paired_path, paired)
    write_jsonl(examples_path, examples)

    print(f"Wrote {len(paired)} paired rollouts -> {paired_path}")
    print(f"Wrote {len(examples)} examples -> {examples_path}")


if __name__ == "__main__":
    main()
