#!/usr/bin/env python3
"""
Vocabulary Projection of Residual Direction
===========================================

Projects residual direction onto the model's unembedding matrix to find
top aligned/anti-aligned tokens.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_invariant_direction(invariant_dir: str, layer: int) -> np.ndarray:
    path = os.path.join(invariant_dir, f"invariant_layer_{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invariant probe not found: {path}")
    state = torch.load(path, map_location="cpu")
    if "classifier.weight" not in state:
        raise ValueError(f"Missing classifier.weight in {path}")
    w = state["classifier.weight"].squeeze().cpu().numpy()
    return w


def select_top_layers(sweep_results: Dict, pooling: str, top_k: int) -> List[int]:
    results = sweep_results.get("results", {}).get(pooling, [])
    scored = []
    for r in results:
        if "error" in r:
            continue
        layer = r.get("layer")
        if layer is None:
            continue
        auc = None
        if "eval_on_insider" in r:
            auc = r["eval_on_insider"].get("invariant_core")
        elif "ood_auc" in r:
            if isinstance(r["ood_auc"], dict):
                auc = r["ood_auc"].get("invariant_core")
            else:
                try:
                    auc = float(r["ood_auc"])
                except Exception:
                    auc = None
        if auc is None:
            continue
        scored.append((auc, int(layer)))
    scored.sort(reverse=True)
    return [layer for _, layer in scored[:top_k]]


def filter_special_tokens(tokenizer, token_ids: List[int]) -> List[int]:
    special_ids = set(tokenizer.all_special_ids)
    return [i for i in token_ids if i not in special_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Project residual direction onto unembedding vocab")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--sweep_results", required=True, help="Path to sweep_results.json")
    parser.add_argument("--invariant_probes_dir", required=True, help="Invariant probes dir (mean pooling)")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--top_k_layers", type=int, default=3)
    parser.add_argument("--top_k_tokens", type=int, default=30)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.sweep_results, "r") as f:
        sweep_results = json.load(f)

    layers = select_top_layers(sweep_results, args.pooling, args.top_k_layers)
    if not layers:
        print("No layers found from sweep_results.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    W = model.get_output_embeddings().weight.detach().cpu().numpy()

    for layer in layers:
        w = load_invariant_direction(args.invariant_probes_dir, layer)
        w = w / (np.linalg.norm(w) + 1e-8)

        if W.shape[1] != w.shape[0]:
            raise ValueError(f"Unembedding dim {W.shape[1]} != residual dim {w.shape[0]}")

        scores = W @ w
        token_ids = list(range(len(scores)))
        token_ids = filter_special_tokens(tokenizer, token_ids)

        token_scores = [(i, scores[i]) for i in token_ids]
        token_scores.sort(key=lambda x: x[1], reverse=True)

        top_pos = token_scores[: args.top_k_tokens]
        top_neg = token_scores[-args.top_k_tokens :][::-1]

        def to_entry(item):
            tid, score = item
            return {
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "score": float(score),
            }

        out = {
            "layer": layer,
            "top_positive": [to_entry(x) for x in top_pos],
            "top_negative": [to_entry(x) for x in top_neg],
        }

        out_json = os.path.join(args.output_dir, f"layer_{layer}_top_tokens.json")
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved: {out_json}")

        # CSV
        out_csv = os.path.join(args.output_dir, f"layer_{layer}_top_tokens.csv")
        with open(out_csv, "w") as f:
            f.write("rank,polarity,token_id,token,score\n")
            for idx, entry in enumerate(out["top_positive"], 1):
                f.write(f"{idx},pos,{entry['token_id']},{entry['token']},{entry['score']}\n")
            for idx, entry in enumerate(out["top_negative"], 1):
                f.write(f"{idx},neg,{entry['token_id']},{entry['token']},{entry['score']}\n")
        print(f"✓ Saved: {out_csv}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        pos_tokens = [e["token"] for e in out["top_positive"]]
        pos_scores = [e["score"] for e in out["top_positive"]]
        neg_tokens = [e["token"] for e in out["top_negative"]]
        neg_scores = [e["score"] for e in out["top_negative"]]

        axes[0].barh(pos_tokens[::-1], pos_scores[::-1], color="#2ecc71")
        axes[0].set_title("Top Positive Tokens")
        axes[0].set_xlabel("Score")

        axes[1].barh(neg_tokens[::-1], neg_scores[::-1], color="#e74c3c")
        axes[1].set_title("Top Negative Tokens")
        axes[1].set_xlabel("Score")

        fig.suptitle(f"Residual Vocabulary Projection (Layer {layer})")
        plt.tight_layout()

        out_png = os.path.join(args.output_dir, f"layer_{layer}_top_tokens.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
