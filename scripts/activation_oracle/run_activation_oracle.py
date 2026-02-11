#!/usr/bin/env python3
"""Run activation-oracle inference jobs with dynamic placeholder injection."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from common import (
    append_jsonl,
    build_run_root,
    ensure_dir,
    get_completed_set,
    init_logger,
    init_status,
    load_progress,
    make_run_id,
    mark_step_completed,
    read_json,
    save_completed_set,
    save_progress,
    update_status,
    utc_now_iso,
    write_json,
)

STEP_SETUP = "setup"
STEP_RUN = "run_jobs"
STEP_FINALIZE = "finalize"


def parse_bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"Invalid bool value: {v}")


def load_jobs(path: Path) -> List[dict]:
    jobs = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            jobs.append(json.loads(line))
    return jobs


def load_existing_response_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                job_id = row.get("job_id")
                if job_id:
                    out.add(job_id)
            except Exception:
                continue
    return out


def get_transformer_blocks(model):
    # Unwrap PEFT models if present.
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model"):
            model = base.model
        else:
            model = base

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for hook insertion")


def build_prompt(target_layer: int, n_vectors: int, question: str, placeholder_text: str) -> str:
    return f"Layer {target_layer}:" + (placeholder_text * n_vectors) + f" {question}"


def run(args: argparse.Namespace) -> int:
    jobs_path = Path(args.jobs_jsonl).resolve()
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
    progress_every = max(1, int(args.progress_every))

    jobs_name = jobs_path.stem
    run_id = args.run_id or make_run_id()
    variant = f"{args.norm_policy}_hook{int(args.hook_layer_index)}"

    run_root = build_run_root(
        output_root=Path(args.output_root).resolve(),
        experiment_name=args.experiment_name,
        model_name=args.ao_model_id,
        dataset_name=jobs_name,
        probe_set="ao_jobs",
        variant=variant,
        run_id=run_id,
    )

    inputs_dir = run_root / "inputs"
    checkpoints_dir = run_root / "checkpoints"
    results_dir = run_root / "results"
    logs_dir = run_root / "logs"
    meta_dir = run_root / "meta"
    for p in (inputs_dir, checkpoints_dir, results_dir, logs_dir, meta_dir):
        ensure_dir(p)

    logger = init_logger(logs_dir / "run.log")
    progress_path = checkpoints_dir / "progress.json"
    progress = load_progress(progress_path)
    completed_steps = set(progress.get("completed_steps", []))

    status_path = meta_dir / "status.json"
    if status_path.exists() and not args.force_rebuild:
        status = read_json(status_path)
        if status.get("state") == "completed":
            logger.info("Status already completed: %s", status_path)
            return 0

    write_json(
        meta_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "generated_at_utc": utc_now_iso(),
            "script": "scripts/activation_oracle/run_activation_oracle.py",
            "jobs_jsonl": str(jobs_path),
            "ao_model_id": args.ao_model_id,
            "base_model_id": args.base_model_id,
            "peft_adapter_id": args.peft_adapter_id,
            "use_fast_tokenizer": bool(args.use_fast_tokenizer),
            "placeholder_text": args.placeholder_text,
            "hook_layer_index": int(args.hook_layer_index),
            "norm_policy": args.norm_policy,
            "alpha": float(args.alpha),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_jobs": int(args.max_jobs),
            "progress_every": int(args.progress_every),
            "force_rebuild": bool(args.force_rebuild),
        },
    )

    init_status(meta_dir, run_id=run_id, state="running", current_step=STEP_SETUP)

    try:
        jobs = load_jobs(jobs_path)
        if not jobs:
            raise ValueError("No jobs in jobs_jsonl")

        write_json(inputs_dir / "job_source.json", {"jobs_jsonl": str(jobs_path), "num_jobs": len(jobs)})

        tokenizer_id = args.ao_model_id
        if args.peft_adapter_id:
            if not args.base_model_id:
                raise ValueError("--base_model_id is required when using --peft_adapter_id")
            tokenizer_id = args.base_model_id
            try:
                from peft import PeftModel
            except Exception as exc:  # pragma: no cover
                raise ImportError("peft is required for adapter loading. Install with `pip install peft`.") from exc

            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_id,
                torch_dtype=getattr(torch, args.torch_dtype),
                device_map=args.device_map,
            )
            model = PeftModel.from_pretrained(base_model, args.peft_adapter_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.ao_model_id,
                torch_dtype=getattr(torch, args.torch_dtype),
                device_map=args.device_map,
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=bool(args.use_fast_tokenizer))
        model.eval()

        placeholder_ids = tokenizer.encode(args.placeholder_text, add_special_tokens=False)
        if len(placeholder_ids) != 1:
            raise ValueError(
                f"Placeholder text must map to a single token. text={args.placeholder_text!r}, ids={placeholder_ids}"
            )
        placeholder_id = int(placeholder_ids[0])

        blocks = get_transformer_blocks(model)
        block_index = int(args.hook_layer_index) - 1
        if block_index < 0 or block_index >= len(blocks):
            raise ValueError(
                f"hook_layer_index={args.hook_layer_index} out of range for model with {len(blocks)} blocks"
            )

        mark_step_completed(progress, STEP_SETUP)
        save_progress(progress_path, progress)
        completed_steps.add(STEP_SETUP)

        update_status(meta_dir, state="running", current_step=STEP_RUN)

        responses_path = results_dir / "responses.jsonl"
        if args.force_rebuild:
            if responses_path.exists():
                responses_path.unlink()
            existing_response_ids = set()
            done_jobs = set()
        else:
            existing_response_ids = load_existing_response_ids(responses_path)
            done_jobs = get_completed_set(progress, "job_units")

        max_jobs = int(args.max_jobs)
        processed = 0
        success = 0
        failed = 0

        model_device = next(model.parameters()).device
        jobs_parent = jobs_path.parent

        runnable_jobs = []
        for job in jobs:
            job_id = str(job.get("job_id", ""))
            if not job_id:
                continue
            if (job_id in done_jobs) or (job_id in existing_response_ids):
                continue
            runnable_jobs.append(job)
        if max_jobs > 0:
            runnable_jobs = runnable_jobs[:max_jobs]

        logger.info(
            "AO run jobs: total=%d runnable=%d skipped_existing=%d",
            len(jobs),
            len(runnable_jobs),
            len(jobs) - len(runnable_jobs),
        )

        jobs_bar = tqdm(runnable_jobs, desc="AO inference jobs", unit="job", leave=True)
        for job in jobs_bar:
            job_id = str(job.get("job_id", ""))
            processed += 1
            row = {
                "job_id": job_id,
                "generated_at_utc": utc_now_iso(),
                "status": "ok",
            }

            try:
                payload_rel = job["vector_payload_path"]
                payload_path = (jobs_parent / payload_rel).resolve()
                if not payload_path.exists():
                    raise FileNotFoundError(f"Missing payload file: {payload_path}")

                with np.load(payload_path, allow_pickle=False) as payload:
                    vectors = payload["vectors"].astype(np.float32)
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
                n_vectors = int(vectors.shape[0])

                question = str(job["question_text"])
                target_layer = int(job["target_layer"])
                prompt = build_prompt(target_layer, n_vectors, question, args.placeholder_text)

                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                input_ids = inputs["input_ids"]
                positions = (input_ids[0] == placeholder_id).nonzero(as_tuple=False).squeeze(-1)
                if int(positions.numel()) != n_vectors:
                    raise ValueError(
                        f"Placeholder count mismatch: found {int(positions.numel())}, expected {n_vectors}"
                    )

                vectors_t = torch.from_numpy(vectors).to(model_device, dtype=next(model.parameters()).dtype)
                done_flag = {"done": False}
                scales_applied: List[float] = []

                def pre_hook(module, module_inputs):
                    if done_flag["done"]:
                        return
                    hidden_states = module_inputs[0]

                    for i, pos in enumerate(positions.tolist()):
                        x = hidden_states[:, pos, :]
                        v = vectors_t[i].unsqueeze(0)
                        if args.norm_policy == "hidden_state_match":
                            x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
                            v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
                            scale = x_norm / v_norm
                            x_new = x + scale * v
                            scales_applied.append(float(scale.mean().item()))
                        else:
                            x_new = x + float(args.alpha) * v
                            scales_applied.append(float(args.alpha))
                        hidden_states[:, pos, :] = x_new

                    done_flag["done"] = True
                    return (hidden_states,) + tuple(module_inputs[1:])

                hook = blocks[block_index].register_forward_pre_hook(pre_hook)
                try:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    )
                finally:
                    hook.remove()

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                row.update(
                    {
                        "experiment": job.get("experiment"),
                        "split": job.get("split"),
                        "pooling": job.get("pooling"),
                        "layer": job.get("layer"),
                        "k": job.get("k"),
                        "pc_index": job.get("pc_index", None),
                        "sample_id": job.get("sample_id"),
                        "label": job.get("label"),
                        "target_layer": target_layer,
                        "prompt_text": prompt,
                        "placeholder_count": n_vectors,
                        "placeholder_positions": [int(p) for p in positions.tolist()],
                        "norm_policy": args.norm_policy,
                        "hook_layer_index": int(args.hook_layer_index),
                        "scales_applied": scales_applied,
                        "generated_text": generated_text,
                    }
                )
                success += 1
            except Exception as e:
                row.update(
                    {
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                failed += 1

            append_jsonl(responses_path, [row])
            existing_response_ids.add(job_id)
            done_jobs.add(job_id)
            jobs_bar.set_postfix({"ok": success, "err": failed})
            if processed % progress_every == 0:
                logger.info("AO progress: processed=%d ok=%d err=%d", processed, success, failed)
            save_completed_set(progress, "job_units", done_jobs)
            save_progress(progress_path, progress)

        jobs_bar.close()

        mark_step_completed(progress, STEP_RUN)
        save_progress(progress_path, progress)

        update_status(meta_dir, state="running", current_step=STEP_FINALIZE)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "run_root": str(run_root),
            "jobs_source": str(jobs_path),
            "num_jobs_in_source": len(jobs),
            "processed_jobs": int(processed),
            "success_jobs": int(success),
            "failed_jobs": int(failed),
            "responses_path": str(responses_path),
        }
        write_json(results_dir / "results.json", summary)

        mark_step_completed(progress, STEP_FINALIZE)
        save_progress(progress_path, progress)

        update_status(meta_dir, state="completed", current_step=None)
        logger.info("Completed AO run. Summary: %s", results_dir / "results.json")
        return 0

    except Exception:
        update_status(meta_dir, state="failed", current_step=None)
        write_json(meta_dir / "last_error.json", {"generated_at_utc": utc_now_iso(), "traceback": traceback.format_exc()})
        logger.exception("AO run failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run activation-oracle jobs")
    p.add_argument("--jobs_jsonl", type=str, required=True)
    p.add_argument("--ao_model_id", type=str, required=True)
    p.add_argument("--base_model_id", type=str, default=None)
    p.add_argument("--peft_adapter_id", type=str, default=None)
    p.add_argument("--use_fast_tokenizer", type=parse_bool, default=True)
    p.add_argument("--placeholder_text", type=str, default=" ?")
    p.add_argument("--hook_layer_index", type=int, default=1)
    p.add_argument("--norm_policy", type=str, default="hidden_state_match", choices=["hidden_state_match", "fixed_alpha"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--do_sample", type=parse_bool, default=False)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_jobs", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=25, help="Emit log progress every N jobs")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--output_root", type=str, default="artifacts")
    p.add_argument("--experiment_name", type=str, default="activation_oracle_inference")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--force_rebuild", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
