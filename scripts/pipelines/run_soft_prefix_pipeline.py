#!/usr/bin/env python3
"""
Clean Soft Prefix Pipeline - End-to-End Training and Evaluation.

This script handles the ENTIRE soft prefix pipeline in one place:
1. Data preparation: Load raw YAML, create train/val/test splits
2. Training: Train soft prefix with validation monitoring
3. Activation caching: Generate activations with learned prefix
4. Evaluation: Compute AUROC on ID val, ID test, OOD test

Usage:
    python scripts/pipelines/run_soft_prefix_pipeline.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --id_raw_yaml data/apollo_raw/roleplaying/dataset.yaml \
        --ood_raw_json data/apollo_raw/insider_trading/llama-70b-3.3-generations.json \
        --probe_path data/probes_layer_agnostic/.../last/probe.pt \
        --layer_idx 15 \
        --output_base /content/drive/MyDrive/checkpoints/soft_prefix_clean

Key fixes over previous implementation:
- No cached manifests - loads fresh from raw YAML/JSON only
- Proper train/val/test splits (60/20/20)
- Validation AUROC computed during training
- Best checkpoint saved based on validation performance
- Clear data flow with explicit paths
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from safetensors.torch import save_file, load_file
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import yaml

# Add actprobe to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: Data Preparation
# ============================================================================

@dataclass
class DataSample:
    """A single data sample."""
    id: str
    text: str
    label: int  # 0=honest, 1=deceptive
    metadata: dict


def load_roleplaying_raw(yaml_path: str) -> List[DataSample]:
    """
    Load Roleplaying dataset from raw YAML.
    
    Expected format:
    - scenario: "..."
      question: "..."
      answer_prefix: "..."
      honest_completion: "..."
      deceptive_completion: "..."
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle both list format and dict with 'data' key
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    
    samples = []
    for i, item in enumerate(data):
        scenario = item.get('scenario', '')
        question = item.get('question', '')
        answer_prefix = item.get('answer_prefix', '')
        
        base_text = f"{scenario}\n{question}\n{answer_prefix}"
        
        # Honest sample
        honest = item.get('honest_completion', '')
        if honest:
            samples.append(DataSample(
                id=f"roleplaying_{i}_honest",
                text=f"{base_text}{honest}".strip(),
                label=0,
                metadata={'scenario_idx': i, 'type': 'honest'}
            ))
        
        # Deceptive sample
        deceptive = item.get('deceptive_completion', '')
        if deceptive:
            samples.append(DataSample(
                id=f"roleplaying_{i}_deceptive",
                text=f"{base_text}{deceptive}".strip(),
                label=1,
                metadata={'scenario_idx': i, 'type': 'deceptive'}
            ))
    
    logger.info(f"Loaded {len(samples)} samples from {yaml_path}")
    return samples


def load_insider_trading_raw(json_path: str) -> List[DataSample]:
    """
    Load InsiderTrading dataset from raw JSON (Apollo format).
    
    JSON format:
    [
        {
            "transcript": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            "metadata": {
                "made_trade": "yes" | "no" | "invalid",
                "trade_message": "..."
            }
        },
        ...
    ]
    
    Label mapping:
    - "yes": Made insider trade → DECEPTIVE (label=1)
    - "no": Refused → HONEST (label=0)
    - "invalid": Skip
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Apollo data is a list of rollouts
    if not isinstance(data, list):
        data = data.get('data', data.get('rollouts', []))
    
    samples = []
    skipped = {'no_label': 0, 'invalid': 0}
    
    LABEL_MAP = {
        "yes": 1,       # Made insider trade - DECEPTIVE
        "no": 0,        # Refused to trade - HONEST
        "invalid": -1   # SKIP
    }
    
    for i, rollout in enumerate(data):
        # Get metadata
        metadata = rollout.get("metadata", {})
        made_trade = metadata.get("made_trade")
        
        # Skip if no trade decision
        if made_trade is None:
            skipped['no_label'] += 1
            continue
        
        # Map label
        label = LABEL_MAP.get(str(made_trade).lower().strip(), -1)
        
        # Skip invalid
        if label == -1:
            skipped['invalid'] += 1
            continue
        
        # Extract transcript and build full text
        transcript = rollout.get("transcript", [])
        
        # Build full context from transcript
        full_context = ""
        for msg in transcript:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                full_context += f"System: {content}\n\n"
            elif role == "user":
                full_context += f"User: {content}\n\n"
            elif role == "assistant":
                full_context += f"Assistant: {content}\n\n"
        
        # If we have the trade_message in metadata, use that as summary
        trade_message = metadata.get("trade_message", "")
        
        # Use full context as text (or fall back to trade_message)
        text = full_context.strip() if full_context else trade_message
        
        if text:
            samples.append(DataSample(
                id=f"insider_{i}",
                text=text,
                label=label,
                metadata={'idx': i, 'made_trade': made_trade}
            ))
    
    logger.info(f"Loaded {len(samples)} samples from {json_path}")
    logger.info(f"  Skipped: {skipped['no_label']} no label, {skipped['invalid']} invalid")
    n_honest = sum(1 for s in samples if s.label == 0)
    n_deceptive = sum(1 for s in samples if s.label == 1)
    logger.info(f"  Labels: {n_honest} honest, {n_deceptive} deceptive")
    
    return samples


def split_data(
    samples: List[DataSample],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[DataSample]]:
    """
    Split data into train/val/test with stratification by label.
    """
    random.seed(seed)
    
    # Separate by label for stratification
    honest = [s for s in samples if s.label == 0]
    deceptive = [s for s in samples if s.label == 1]
    
    random.shuffle(honest)
    random.shuffle(deceptive)
    
    def split_list(lst):
        n = len(lst)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    h_train, h_val, h_test = split_list(honest)
    d_train, d_val, d_test = split_list(deceptive)
    
    splits = {
        'train': h_train + d_train,
        'val': h_val + d_val,
        'test': h_test + d_test
    }
    
    # Shuffle each split
    for key in splits:
        random.shuffle(splits[key])
    
    logger.info(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    for key, data in splits.items():
        n_honest = sum(1 for s in data if s.label == 0)
        n_deceptive = sum(1 for s in data if s.label == 1)
        logger.info(f"  {key}: {n_honest} honest, {n_deceptive} deceptive")
    
    return splits


def save_splits_jsonl(splits: Dict[str, List[DataSample]], output_dir: str):
    """Save splits as JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, samples in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, 'w') as f:
            for s in samples:
                f.write(json.dumps({
                    'id': s.id,
                    'model_input': s.text,
                    'label': s.label,
                    'metadata': s.metadata
                }) + '\n')
        logger.info(f"Saved {len(samples)} samples to {path}")


# ============================================================================
# Phase 2: Soft Prefix Training
# ============================================================================

class SimpleDataset(Dataset):
    """Simple dataset from list of DataSamples."""
    
    def __init__(self, samples: List[DataSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.text, s.label
    
    def get_labels(self):
        return [s.label for s in self.samples]


class PairDataset(Dataset):
    """Dataset of (honest, deceptive) pairs grouped by scenario_idx."""

    def __init__(self, samples: List[DataSample]):
        pairs = {}
        for sample in samples:
            scenario_idx = sample.metadata.get('scenario_idx')
            if scenario_idx is None:
                continue
            if scenario_idx not in pairs:
                pairs[scenario_idx] = {}
            pairs[scenario_idx][sample.metadata.get('type')] = sample

        self.pairs = []
        for scenario_idx, items in pairs.items():
            honest = items.get('honest')
            deceptive = items.get('deceptive')
            if honest is None or deceptive is None:
                continue
            self.pairs.append((honest, deceptive))

        logger.info(f"Constructed {len(self.pairs)} paired samples from {len(samples)} inputs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        honest, deceptive = self.pairs[idx]
        return honest.text, deceptive.text


class BalancedBatchSampler(Sampler):
    """Yields batches with B/2 honest, B/2 deceptive."""
    
    def __init__(self, labels: List[int], batch_size: int, shuffle: bool = True):
        self.pos_indices = [i for i, l in enumerate(labels) if l == 1]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0]
        self.half = batch_size // 2
        self.shuffle = shuffle
    
    def __iter__(self):
        pos = self.pos_indices.copy()
        neg = self.neg_indices.copy()
        
        if self.shuffle:
            random.shuffle(pos)
            random.shuffle(neg)
        
        min_len = min(len(pos), len(neg))
        for i in range(0, min_len, self.half):
            pos_batch = pos[i:i+self.half]
            neg_batch = neg[i:i+self.half]
            if len(pos_batch) == self.half and len(neg_batch) == self.half:
                yield pos_batch + neg_batch
    
    def __len__(self):
        min_len = min(len(self.pos_indices), len(self.neg_indices))
        return min_len // self.half


class SoftPrefixWrapper(nn.Module):
    """Wraps model to prepend learnable soft prefix embeddings."""
    
    def __init__(self, model, tokenizer, prompt_len: int):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        hidden_dim = model.config.hidden_size
        
        # Freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Learnable prefix - small init
        self.soft_prefix = nn.Parameter(torch.randn(1, prompt_len, hidden_dim) * 0.01)
        
        logger.info(f"SoftPrefixWrapper: prompt_len={prompt_len}, hidden_dim={hidden_dim}")
    
    def forward(self, texts: List[str]):
        """Forward pass with soft prefix prepended."""
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(self.model.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        B, T = input_ids.shape
        
        # Token embeddings
        token_embeds = self.model.model.embed_tokens(input_ids)
        
        # Expand prefix
        prefix_embeds = self.soft_prefix.to(device=token_embeds.device, dtype=token_embeds.dtype)
        prefix_embeds = prefix_embeds.expand(B, -1, -1)
        
        # Concat
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        P = self.prompt_len
        prefix_mask = torch.ones(B, P, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        
        return outputs.hidden_states, combined_mask
    
    def get_final_token_features(self, texts: List[str], layer_idx: int) -> torch.Tensor:
        """Extract hidden state at final token for specified layer."""
        hidden_states, combined_mask = self.forward(texts)
        layer_hidden = hidden_states[layer_idx + 1]  # +1 for embeddings
        
        features = []
        for i in range(len(texts)):
            real_len = combined_mask[i].sum().item()
            final_idx = int(real_len) - 1
            features.append(layer_hidden[i, final_idx, :])
        
        return torch.stack(features)


def contrastive_hinge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_prefix: nn.Parameter,
    margin: float = 1.0,
    lambda_norm: float = 1e-3,
    lambda_anchor: float = 1.0,
    safe_threshold: float = -1.0,  # Only penalize if honest rises above this
) -> Tuple[torch.Tensor, dict]:
    """
    Contrastive loss with ONE-SIDED HINGE for honest samples.
    
    Loss = L_contrast + λ_norm * ||prefix|| + λ_anchor * L_hinge
    
    Where:
    - L_contrast = max(0, margin - (μ_dec - μ_hon)) - push deceptive high
    - L_hinge = max(0, μ_hon - safe_threshold)² - ONLY penalize if honest 
      drifts UP toward ambiguity/deception zone
    
    KEY INSIGHT from baseline stats:
    - Honest baseline = -1.334 (naturally negative)
    - Sigmoid(0) = 0.5 = AMBIGUOUS (bad!)
    - Sigmoid(-1) = 0.27 = leaning honest
    - Sigmoid(-2) = 0.12 = safely honest
    
    We want honest samples to stay NEGATIVE (safe). If the prefix makes them
    even more negative (-5.0), that's GREAT! Don't penalize that.
    Only penalize if they drift UPWARD toward 0 (ambiguity).
    
    This is like saying: "Keep the furniture away from the fire (-1.0)."
    If it's at -5.0 (very safe), leave it alone. Only pull it back if it
    approaches the danger zone.
    """
    z_dec = logits[labels == 1]
    z_hon = logits[labels == 0]
    
    if len(z_dec) == 0 or len(z_hon) == 0:
        raise ValueError("Batch must contain both classes")
    
    mu_dec = z_dec.mean()
    mu_hon = z_hon.mean()
    separation = mu_dec - mu_hon
    
    # Main contrastive loss - push deceptive above honest by margin
    L_contrast = F.relu(margin - separation)
    
    # Norm regularization on prefix
    L_norm = soft_prefix.norm()
    
    # HINGE ANCHOR: Only penalize if honest drifts UP toward danger zone
    # If mu_hon < safe_threshold, loss = 0 (good, it's safely negative)
    # If mu_hon > safe_threshold, loss = (mu_hon - safe_threshold)²
    L_hinge = F.relu(mu_hon - safe_threshold) ** 2
    
    loss = L_contrast + lambda_norm * L_norm + lambda_anchor * L_hinge
    
    return loss, {
        'mu_dec': mu_dec.item(),
        'mu_hon': mu_hon.item(),
        'separation': separation.item(),
        'L_contrast': L_contrast.item(),
        'L_hinge': L_hinge.item(),
        'loss': loss.item()
    }


def pairwise_margin_loss(
    honest_logits: torch.Tensor,
    deceptive_logits: torch.Tensor,
    soft_prefix: nn.Parameter,
    margin: float = 1.0,
    lambda_norm: float = 1e-3
) -> Tuple[torch.Tensor, dict]:
    """
    Pairwise margin loss on paired honest/deceptive completions.

    Loss = mean(max(0, margin - (z_dec - z_hon))) + λ_norm * ||prefix||
    """
    deltas = deceptive_logits - honest_logits
    pair_loss = F.relu(margin - deltas)
    L_norm = soft_prefix.norm()
    loss = pair_loss.mean() + lambda_norm * L_norm
    pair_acc = (deltas > 0).float().mean()

    return loss, {
        'pair_margin': deltas.mean().item(),
        'pair_acc': pair_acc.item(),
        'L_pair': pair_loss.mean().item(),
        'loss': loss.item()
    }


def compute_baseline_logits(
    model,
    tokenizer,
    probe: nn.Module,
    samples: List[DataSample],
    layer_idx: int,
    batch_size: int = 4
) -> Dict[str, float]:
    """
    Compute baseline logits WITHOUT prefix for anchoring.
    
    Returns dict mapping sample_id -> baseline_logit
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    probe = probe.to(device=device, dtype=dtype)
    probe.eval()
    model.eval()
    
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    baseline = {}
    
    logger.info("Computing baseline logits (no prefix)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Baseline"):
            batch = samples[i:i+batch_size]
            texts = [s.text for s in batch]
            
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048
            ).to(device)
            
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
            # Get final token features at target layer
            layer_hidden = outputs.hidden_states[layer_idx + 1]
            
            for j, sample in enumerate(batch):
                real_len = inputs.attention_mask[j].sum().item()
                final_idx = int(real_len) - 1
                features = layer_hidden[j, final_idx, :].unsqueeze(0).unsqueeze(1)
                logit = probe(features).squeeze().item()
                baseline[sample.id] = logit
    
    # Log baseline stats
    honest_baselines = [baseline[s.id] for s in samples if s.label == 0]
    deceptive_baselines = [baseline[s.id] for s in samples if s.label == 1]
    
    logger.info(f"Baseline stats:")
    logger.info(f"  Honest:    mean={np.mean(honest_baselines):.3f}, std={np.std(honest_baselines):.3f}")
    logger.info(f"  Deceptive: mean={np.mean(deceptive_baselines):.3f}, std={np.std(deceptive_baselines):.3f}")
    logger.info(f"  Separation: {np.mean(deceptive_baselines) - np.mean(honest_baselines):.3f}")
    
    return baseline


def determine_probe_sign(
    model,
    tokenizer,
    probe: nn.Module,
    samples: List[DataSample],
    layer_idx: int,
    batch_size: int = 4
) -> float:
    """
    Determine whether probe logits are oriented correctly.

    Returns +1.0 if deceptive mean > honest mean, else -1.0 to flip logits.
    """
    baseline = compute_baseline_logits(
        model, tokenizer, probe, samples, layer_idx, batch_size=batch_size
    )
    honest_baselines = [baseline[s.id] for s in samples if s.label == 0]
    deceptive_baselines = [baseline[s.id] for s in samples if s.label == 1]
    separation = np.mean(deceptive_baselines) - np.mean(honest_baselines)
    logit_sign = 1.0 if separation >= 0 else -1.0
    logger.info(
        f"Probe orientation: separation={separation:.3f} -> logit_sign={logit_sign:+.0f}"
    )
    return logit_sign


def compute_validation_auroc(
    wrapper: SoftPrefixWrapper,
    probe: nn.Module,
    val_samples: List[DataSample],
    layer_idx: int,
    batch_size: int = 8,
    logit_sign: float = 1.0
) -> float:
    """Compute AUROC on validation set."""
    wrapper.eval()
    probe.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i+batch_size]
            texts = [s.text for s in batch]
            labels = [s.label for s in batch]
            
            features = wrapper.get_final_token_features(texts, layer_idx)
            
            # For layer-agnostic probes, features are already (B, D)
            # No token dimension needed
            logits = probe(features.unsqueeze(1)) * logit_sign  # Add token dim for probe
            
            all_logits.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(labels)
    
    try:
        auroc = roc_auc_score(all_labels, all_logits)
    except:
        auroc = 0.5
    
    return auroc


def train_soft_prefix(
    wrapper: SoftPrefixWrapper,
    probe: nn.Module,
    train_samples: List[DataSample],
    val_samples: List[DataSample],
    layer_idx: int,
    args,
    logit_sign: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Train soft prefix with validation monitoring.
    
    Uses contrastive loss with honest anchoring to TARGET=0.
    This ensures the prefix improves separation WITHOUT destroying the probe.
    
    Returns:
        best_prefix: Best prefix tensor
        training_log: Dict with metrics per step
    """
    device = next(wrapper.model.parameters()).device
    dtype = next(wrapper.model.parameters()).dtype
    probe = probe.to(device=device, dtype=dtype)
    
    # Setup data
    if args.loss_type == "pairwise":
        train_dataset = PairDataset(train_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: ([h for h, _ in b], [d for _, d in b])
        )
    else:
        train_dataset = SimpleDataset(train_samples)
        sampler = BalancedBatchSampler(train_dataset.get_labels(), args.batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: ([t for t, l in b], torch.tensor([l for t, l in b]))
        )
    
    optimizer = torch.optim.AdamW([wrapper.soft_prefix], lr=args.lr)
    
    best_auroc = 0.0
    best_prefix = None
    training_log = {
        'steps': [],
        'train_loss': [],
        'val_auroc': [],
        'separation': [],
        'L_hinge': [],
        'mu_hon': [],
        'mu_dec': [],
        'pair_margin': [],
        'pair_acc': []
    }
    
    step = 0
    pbar = tqdm(total=args.steps, desc="Training")
    
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            if args.loss_type == "pairwise":
                honest_texts, deceptive_texts = batch
                texts = honest_texts + deceptive_texts
                features = wrapper.get_final_token_features(texts, layer_idx)
                split = len(honest_texts)
                honest_features = features[:split]
                deceptive_features = features[split:]

                honest_logits = probe(honest_features.unsqueeze(1)).squeeze() * logit_sign
                deceptive_logits = probe(deceptive_features.unsqueeze(1)).squeeze() * logit_sign

                loss, metrics = pairwise_margin_loss(
                    honest_logits, deceptive_logits, wrapper.soft_prefix,
                    margin=args.margin, lambda_norm=args.lambda_norm
                )
            else:
                texts, labels = batch
                labels = labels.to(device)

                # Forward with prefix
                features = wrapper.get_final_token_features(texts, layer_idx)
                logits = probe(features.unsqueeze(1)).squeeze() * logit_sign

                # Hinge loss: only penalize if honest drifts UP toward ambiguity
                loss, metrics = contrastive_hinge_loss(
                    logits, labels, wrapper.soft_prefix,
                    margin=args.margin, lambda_norm=args.lambda_norm,
                    lambda_anchor=args.lambda_anchor, safe_threshold=args.safe_threshold
                )
            
            labels = labels.to(device)
            
            # Forward with prefix
            features = wrapper.get_final_token_features(texts, layer_idx)
            logits = probe(features.unsqueeze(1)).squeeze() * logit_sign
            
            # Hinge loss: only penalize if honest drifts UP toward ambiguity
            loss, metrics = contrastive_hinge_loss(
                logits, labels, wrapper.soft_prefix,
                margin=args.margin, lambda_norm=args.lambda_norm,
                lambda_anchor=args.lambda_anchor, safe_threshold=args.safe_threshold
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.update(1)
            if args.loss_type == "pairwise":
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    pair_margin=f"{metrics['pair_margin']:.2f}",
                    pair_acc=f"{metrics['pair_acc']:.2f}"
                )
            else:
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    sep=f"{metrics['separation']:.2f}",
                    mu_h=f"{metrics['mu_hon']:.2f}",
                    hinge=f"{metrics['L_hinge']:.3f}"
                )
            
            # Validation every N steps
            if step % args.val_every == 0:
                val_auroc = compute_validation_auroc(
                    wrapper, probe, val_samples, layer_idx, args.batch_size, logit_sign
                )
                
                training_log['steps'].append(step)
                training_log['train_loss'].append(metrics['loss'])
                training_log['val_auroc'].append(val_auroc)

                if args.loss_type == "pairwise":
                    training_log['pair_margin'].append(metrics['pair_margin'])
                    training_log['pair_acc'].append(metrics['pair_acc'])
                    logger.info(
                        f"Step {step}: val_auroc={val_auroc:.4f}, "
                        f"pair_margin={metrics['pair_margin']:.2f}, "
                        f"pair_acc={metrics['pair_acc']:.2f}"
                    )
                else:
                    training_log['separation'].append(metrics['separation'])
                    training_log['L_hinge'].append(metrics['L_hinge'])
                    training_log['mu_hon'].append(metrics['mu_hon'])
                    training_log['mu_dec'].append(metrics['mu_dec'])
                    logger.info(
                        f"Step {step}: val_auroc={val_auroc:.4f}, "
                        f"sep={metrics['separation']:.2f}, "
                        f"μ_hon={metrics['mu_hon']:.2f}, "
                        f"L_hinge={metrics['L_hinge']:.3f}"
                    )
                
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_prefix = wrapper.soft_prefix.detach().clone()
                    logger.info(f"  New best! auroc={val_auroc:.4f}")
    
    pbar.close()
    
    training_log['best_auroc'] = best_auroc
    
    return best_prefix, training_log


# ============================================================================
# Phase 3: Activation Caching
# ============================================================================

def cache_activations_with_prefix(
    model,
    tokenizer,
    soft_prefix: torch.Tensor,
    samples: List[DataSample],
    output_dir: str,
    batch_size: int = 4
):
    """Cache activations using trained soft prefix."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    soft_prefix = soft_prefix.to(device=device, dtype=dtype)
    P = soft_prefix.shape[1] if soft_prefix.dim() == 3 else soft_prefix.shape[0]
    
    # If prefix is [P, D], add batch dim
    if soft_prefix.dim() == 2:
        soft_prefix = soft_prefix.unsqueeze(0)
    
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    SHARD_SIZE = 100
    buffer = {}
    shard_idx = 0
    manifest = []
    
    pbar = tqdm(total=len(samples), desc="Caching activations")
    
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start+batch_size]
        texts = [s.text for s in batch]
        
        with torch.no_grad():
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048
            ).to(device)
            
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            B, T = input_ids.shape
            
            token_embeds = model.model.embed_tokens(input_ids)
            prefix_embeds = soft_prefix.expand(B, -1, -1)
            combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            
            prefix_mask = torch.ones(B, P, device=device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            outputs = model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                output_hidden_states=True
            )
            
            # Get final token activations for each layer
            hs_tuple = outputs.hidden_states[1:]  # Skip embeddings
            stacked = torch.stack(hs_tuple, dim=1)  # [B, L, P+T, D]
            
            for i, sample in enumerate(batch):
                real_len = combined_mask[i].sum().item()
                final_idx = int(real_len) - 1
                acts = stacked[i, :, final_idx, :].cpu()  # [L, D]
                
                buffer[sample.id] = acts
                manifest.append({
                    'id': sample.id,
                    'label': sample.label,
                    'shard': shard_idx,
                    'shape': list(acts.shape)
                })
        
        pbar.update(len(batch))
        
        # Save shard
        if len(buffer) >= SHARD_SIZE:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.safetensors")
            save_file(buffer, shard_path)
            buffer = {}
            shard_idx += 1
    
    # Save remaining
    if buffer:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.safetensors")
        save_file(buffer, shard_path)
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + '\n')
    
    pbar.close()
    logger.info(f"Cached {len(manifest)} samples to {output_dir}")


# ============================================================================
# Phase 4: Evaluation
# ============================================================================

def evaluate_probe_on_activations(
    activations_dir: str,
    probe_path: str,
    layer_idx: int,
    pooling: str = "last",
    logit_sign: float = 1.0
) -> dict:
    """Evaluate probe on cached activations."""
    from actprobe.probes.models import LayerProbe
    
    # Load manifest
    manifest_path = os.path.join(activations_dir, "manifest.jsonl")
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Load activations
    import glob
    shard_files = sorted(glob.glob(os.path.join(activations_dir, "shard_*.safetensors")))
    
    all_acts = {}
    for shard_path in shard_files:
        shard_data = load_file(shard_path)
        all_acts.update({k: v.numpy() for k, v in shard_data.items()})
    
    # Load probe
    hidden_dim = list(all_acts.values())[0].shape[1]  # [L, D] -> D
    probe = LayerProbe(input_dim=hidden_dim, pooling_type=pooling)
    probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
    probe.eval()
    
    # Evaluate
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for sample in samples:
            acts = all_acts[sample['id']]  # [L, D]
            layer_acts = acts[layer_idx]  # [D]
            
            x = torch.tensor(layer_acts).unsqueeze(0).unsqueeze(1).float()  # [1, 1, D]
            logit = probe(x).item() * logit_sign
            
            all_logits.append(torch.sigmoid(torch.tensor(logit)).item())
            all_labels.append(sample['label'])
    
    auroc = roc_auc_score(all_labels, all_logits)
    
    n_honest = sum(1 for l in all_labels if l == 0)
    n_deceptive = sum(1 for l in all_labels if l == 1)
    
    return {
        'auroc': auroc,
        'n_samples': len(samples),
        'n_honest': n_honest,
        'n_deceptive': n_deceptive,
        'layer': layer_idx
    }


# ============================================================================
# Main Orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean Soft Prefix Pipeline")
    
    # Model
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--hf_token", type=str, default=None)
    
    # Data
    parser.add_argument("--id_raw_yaml", type=str, required=True,
                       help="Path to ID raw YAML (roleplaying)")
    parser.add_argument("--ood_raw_json", type=str, required=True,
                       help="Path to OOD raw JSON (insider trading)")
    
    # Probe
    parser.add_argument("--probe_path", type=str, required=True,
                       help="Path to probe checkpoint")
    parser.add_argument("--layer_idx", type=int, default=15)
    parser.add_argument("--pooling", type=str, default="last")
    
    # Training
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_type", type=str, default="mean",
                       choices=["mean", "pairwise"],
                       help="Loss type: mean (batch mean separation) or pairwise (paired ranking)")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for loss (mean separation or pairwise ranking)")
    parser.add_argument("--lambda_norm", type=float, default=1e-3,
                       help="L2 regularization on prefix norm")
    parser.add_argument("--lambda_anchor", type=float, default=1.0,
                       help="Weight for hinge loss on honest logits")
    parser.add_argument("--safe_threshold", type=float, default=-1.0,
                       help="Only penalize if honest rises above this (hinge)")
    parser.add_argument("--val_every", type=int, default=50)
    
    # OOD
    parser.add_argument("--ood_limit", type=int, default=300,
                       help="Max OOD samples to use (default 300)")
    parser.add_argument("--cache_batch_size", type=int, default=2,
                       help="Batch size for caching (smaller to avoid OOM)")
    
    # Output
    parser.add_argument("--output_base", type=str, required=True,
                       help="Base output directory")
    
    # Phases
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, use existing prefix")
    parser.add_argument("--prefix_ckpt", type=str, default=None,
                       help="Path to existing prefix checkpoint (for skip_training)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CLEAN SOFT PREFIX PIPELINE")
    logger.info("=" * 80)
    
    # Setup output dirs
    prefix_dir = os.path.join(args.output_base, "prefix")
    splits_dir = os.path.join(args.output_base, "splits")
    acts_dir = os.path.join(args.output_base, "activations")
    results_dir = os.path.join(args.output_base, "results")
    
    os.makedirs(args.output_base, exist_ok=True)
    
    # =========================================================================
    # Phase 1: Data Preparation
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 40)
    
    # Load ID data
    logger.info(f"\nLoading ID data from: {args.id_raw_yaml}")
    id_samples = load_roleplaying_raw(args.id_raw_yaml)
    
    # Split ID data
    id_splits = split_data(id_samples, train_ratio=0.6, val_ratio=0.2)
    save_splits_jsonl(id_splits, os.path.join(splits_dir, "id"))
    
    # Load OOD data and SAMPLE to limit
    logger.info(f"\nLoading OOD data from: {args.ood_raw_json}")
    ood_samples_all = load_insider_trading_raw(args.ood_raw_json)
    
    # Sample OOD to limit (balanced by class)
    if len(ood_samples_all) > args.ood_limit:
        random.seed(42)
        ood_honest = [s for s in ood_samples_all if s.label == 0]
        ood_deceptive = [s for s in ood_samples_all if s.label == 1]
        random.shuffle(ood_honest)
        random.shuffle(ood_deceptive)
        
        # Take balanced sample
        half = args.ood_limit // 2
        ood_samples = ood_honest[:half] + ood_deceptive[:half]
        random.shuffle(ood_samples)
        
        logger.info(f"Sampled OOD: {len(ood_samples)} from {len(ood_samples_all)} (limit={args.ood_limit})")
    else:
        ood_samples = ood_samples_all
    
    n_ood_honest = sum(1 for s in ood_samples if s.label == 0)
    n_ood_deceptive = sum(1 for s in ood_samples if s.label == 1)
    logger.info(f"OOD final: {len(ood_samples)} samples ({n_ood_honest} honest, {n_ood_deceptive} deceptive)")
    
    # =========================================================================
    # Phase 2: Training
    # =========================================================================
    if not args.skip_training:
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 2: Training Soft Prefix")
        logger.info("=" * 40)
        
        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model, token=args.hf_token,
            torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        
        # Enable gradient checkpointing to reduce memory usage
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for memory efficiency")
        
        # Load probe
        from actprobe.probes.models import LayerProbe
        hidden_dim = model.config.hidden_size
        probe = LayerProbe(input_dim=hidden_dim, pooling_type=args.pooling)
        probe.load_state_dict(torch.load(args.probe_path, map_location='cpu'))
        probe.eval()
        for p in probe.parameters():
            p.requires_grad = False
        
        logger.info(f"Loaded probe from {args.probe_path}")
        
        logit_sign = determine_probe_sign(
            model,
            tokenizer,
            probe,
            id_splits['train'],
            args.layer_idx,
            batch_size=args.batch_size
        )

        # Create wrapper
        wrapper = SoftPrefixWrapper(model, tokenizer, args.prompt_len)
        
        # Train with honest anchoring to 0
        best_prefix, training_log = train_soft_prefix(
            wrapper, probe,
            id_splits['train'], id_splits['val'],
            args.layer_idx, args, logit_sign
        )
        
        # Save
        os.makedirs(prefix_dir, exist_ok=True)
        prefix_2d = best_prefix.squeeze(0)
        save_file({'soft_prefix': prefix_2d}, os.path.join(prefix_dir, 'prefix.safetensors'))
        
        with open(os.path.join(prefix_dir, 'config.json'), 'w') as f:
            json.dump({
                'model': args.model,
                'layer_idx': args.layer_idx,
                'prompt_len': args.prompt_len,
                'steps': args.steps,
                'best_val_auroc': training_log['best_auroc'],
                'logit_sign': logit_sign,
                'loss_type': args.loss_type
            }, f, indent=2)
        
        with open(os.path.join(prefix_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        logger.info(f"\nSaved prefix to {prefix_dir}")
        logger.info(f"Best validation AUROC: {training_log['best_auroc']:.4f}")
        
        soft_prefix = prefix_2d
    else:
        # Load existing prefix
        logger.info(f"\nSkipping training, loading prefix from: {args.prefix_ckpt}")
        tensors = load_file(os.path.join(args.prefix_ckpt, 'prefix.safetensors'))
        soft_prefix = tensors['soft_prefix']
        
        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, token=args.hf_token,
            torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        from actprobe.probes.models import LayerProbe
        hidden_dim = model.config.hidden_size
        probe = LayerProbe(input_dim=hidden_dim, pooling_type=args.pooling)
        probe.load_state_dict(torch.load(args.probe_path, map_location='cpu'))
        probe.eval()
        for p in probe.parameters():
            p.requires_grad = False

        config_path = os.path.join(args.prefix_ckpt, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logit_sign = config.get('logit_sign', 1.0)
            logger.info(f"Loaded logit_sign from config: {logit_sign:+.0f}")
        else:
            logit_sign = determine_probe_sign(
                model,
                tokenizer,
                probe,
                id_splits['train'],
                args.layer_idx,
                batch_size=args.batch_size
            )
    
    # =========================================================================
    # Phase 3: Cache Activations
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 3: Cache Activations with Prefix")
    logger.info("=" * 40)
    
    # ID Validation
    id_val_acts_dir = os.path.join(acts_dir, "id_val")
    logger.info(f"\nCaching ID validation activations...")
    cache_activations_with_prefix(
        model, tokenizer, soft_prefix,
        id_splits['val'], id_val_acts_dir,
        batch_size=args.cache_batch_size
    )
    
    # ID Test
    id_test_acts_dir = os.path.join(acts_dir, "id_test")
    logger.info(f"\nCaching ID test activations...")
    cache_activations_with_prefix(
        model, tokenizer, soft_prefix,
        id_splits['test'], id_test_acts_dir,
        batch_size=args.cache_batch_size
    )
    
    # OOD Test (use even smaller batch for long transcripts)
    ood_test_acts_dir = os.path.join(acts_dir, "ood_test")
    logger.info(f"\nCaching OOD test activations ({len(ood_samples)} samples)...")
    cache_activations_with_prefix(
        model, tokenizer, soft_prefix,
        ood_samples, ood_test_acts_dir,
        batch_size=1  # OOD has long transcripts, use batch=1 to avoid OOM
    )
    
    # =========================================================================
    # Phase 4: Evaluation
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 4: Evaluate Probes")
    logger.info("=" * 40)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # ID Val
    logger.info("\nEvaluating on ID Validation...")
    id_val_results = evaluate_probe_on_activations(
        id_val_acts_dir, args.probe_path, args.layer_idx, args.pooling, logit_sign
    )
    logger.info(f"  ID Val AUROC: {id_val_results['auroc']:.4f}")
    
    # ID Test
    logger.info("\nEvaluating on ID Test...")
    id_test_results = evaluate_probe_on_activations(
        id_test_acts_dir, args.probe_path, args.layer_idx, args.pooling, logit_sign
    )
    logger.info(f"  ID Test AUROC: {id_test_results['auroc']:.4f}")
    
    # OOD Test
    logger.info("\nEvaluating on OOD Test...")
    ood_test_results = evaluate_probe_on_activations(
        ood_test_acts_dir, args.probe_path, args.layer_idx, args.pooling, logit_sign
    )
    logger.info(f"  OOD Test AUROC: {ood_test_results['auroc']:.4f}")
    
    # Save results
    all_results = {
        'id_val': id_val_results,
        'id_test': id_test_results,
        'ood_test': ood_test_results,
        'config': {
            'model': args.model,
            'layer_idx': args.layer_idx,
            'prompt_len': args.prompt_len,
            'loss_type': args.loss_type
        }
    }
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults:")
    logger.info(f"  ID Val AUROC:  {id_val_results['auroc']:.4f} ({id_val_results['n_samples']} samples)")
    logger.info(f"  ID Test AUROC: {id_test_results['auroc']:.4f} ({id_test_results['n_samples']} samples)")
    logger.info(f"  OOD Test AUROC: {ood_test_results['auroc']:.4f} ({ood_test_results['n_samples']} samples)")
    logger.info(f"\nOutput: {args.output_base}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
