#!/usr/bin/env python3
"""
Create an alias activation dataset by copying (or hardlinking) splits/shards.

Example:
  python scripts/data/alias_activation_dataset.py \
    --activations_root /content/drive/MyDrive/Efficacy-of-ensemble-of-attention-probes/data/activations_fullprompt \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --src_dataset Deception-InsiderTrading \
    --dst_dataset Deception-InsiderTrading-full
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


def split_root_and_model(root: Path, model_dir: str) -> Tuple[Path, Path]:
    if root.name == model_dir:
        return root.parent, root
    candidate = root / model_dir
    if candidate.exists():
        return root, candidate
    return root.parent, root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alias activation dataset by copying shards.")
    p.add_argument("--activations_root", type=str, required=True)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--src_dataset", type=str, required=True)
    p.add_argument("--dst_dataset", type=str, required=True)
    p.add_argument("--splits", type=str, default="train,validation,test,val")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying (may not work on Drive).",
    )
    return p.parse_args()


def copy_or_link(src: Path, dst: Path, hardlink: bool) -> None:
    if hardlink:
        if dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.link_to(src)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    model_dir = args.model.replace("/", "_")
    _, acts_model_root = split_root_and_model(Path(args.activations_root), model_dir)
    src_root = acts_model_root / args.src_dataset
    dst_root = acts_model_root / args.dst_dataset

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    copied_any = False
    for split in splits:
        src_split = src_root / split
        if not src_split.exists():
            continue
        dst_split = dst_root / split
        if dst_split.exists():
            if args.force:
                shutil.rmtree(dst_split)
            else:
                print(f"[skip] {dst_split} exists (use --force to overwrite)")
                continue
        dst_split.mkdir(parents=True, exist_ok=True)
        manifest = src_split / "manifest.jsonl"
        if manifest.exists():
            copy_or_link(manifest, dst_split / "manifest.jsonl", args.hardlink)
        for shard in sorted(src_split.glob("shard_*.safetensors")):
            copy_or_link(shard, dst_split / shard.name, args.hardlink)
        copied_any = True
        print(f"[done] {args.src_dataset}/{split} -> {args.dst_dataset}/{split}")

    if not copied_any:
        print("[warn] no splits copied (check src splits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
