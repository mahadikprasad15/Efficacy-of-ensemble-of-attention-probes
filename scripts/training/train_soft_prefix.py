"""
Train a Learnable Soft-Prefix for Improved OOD Deception Monitoring.

This script trains a soft prefix (embedding-level virtual tokens) that is 
prepended to model inputs to improve separation in frozen probe logits 
between deceptive vs honest examples.

Key constraints:
- Prefix is PREPENDED (decoder-only requires past tokens for attention)
- Base LLM is frozen, probes are frozen
- Only soft_prefix embeddings are trained
- Loss is contrastive separation + regularization (not "maximize deception")

Usage:
    python scripts/training/train_soft_prefix.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --train_jsonl data/prepared_datasets/.../train.jsonl \
        --probe_path data/probes/.../probe_layer_15.pt \
        --layer_idx 15 \
        --prompt_len 16 \
        --steps 5000 \
        --out_dir checkpoints/soft_prefix
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from safetensors.torch import save_file
from tqdm import tqdm

# Add actprobe to path
sys.path.append(os.path.join(os.getcwd(), 'actprobe', 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global flag for raw mode (no suffix/Answer: check)
RAW_MODE = False


# ============================================================================
# Dataset + Collate
# ============================================================================

class PreparedDataset(Dataset):
    """Load prepared JSONL for soft prefix training."""
    
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                self.items.append((obj['model_input'], obj['label']))
        logger.info(f"Loaded {len(self.items)} examples from {jsonl_path}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]  # (text, label)
    
    def get_labels(self) -> List[int]:
        return [label for _, label in self.items]


def collate_fn(batch):
    """Custom collate: return list of texts + tensor of labels."""
    texts = [t for t, l in batch]
    labels = torch.tensor([l for t, l in batch], dtype=torch.long)
    return texts, labels


class RawDataset(Dataset):
    """
    Load raw YAML data (without suffix) for soft prefix training.
    
    Expected format (like roleplaying/dataset.yaml):
    - scenario: "..."
      question: "..."
      answer_prefix: "..."
      honest_completion: "..."
      deceptive_completion: "..."
    
    Produces TEXT = scenario + question + answer_prefix + completion
    """
    
    def __init__(self, yaml_path: str):
        import yaml
        self.items = []
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle both list format and dict with 'data' key
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        for i, item in enumerate(data):
            scenario = item.get('scenario', '')
            question = item.get('question', '')
            answer_prefix = item.get('answer_prefix', '')
            
            # Combine scenario + question + answer_prefix as the base
            base_text = f"{scenario}\n{question}\n{answer_prefix}"
            
            # Honest sample
            honest = item.get('honest_completion', '')
            if honest:
                text = f"{base_text}{honest}"
                self.items.append((text.strip(), 0))  # label=0 for honest
            
            # Deceptive sample
            deceptive = item.get('deceptive_completion', '')
            if deceptive:
                text = f"{base_text}{deceptive}"
                self.items.append((text.strip(), 1))  # label=1 for deceptive
        
        logger.info(f"Loaded {len(self.items)} examples from {yaml_path}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def get_labels(self) -> List[int]:
        return [label for _, label in self.items]


# ============================================================================
# Balanced Batch Sampler
# ============================================================================

class BalancedBatchSampler(Sampler):
    """
    Yields batches with B/2 honest (label=0), B/2 deceptive (label=1) samples.
    Required for contrastive separation loss.
    """
    
    def __init__(self, labels: List[int], batch_size: int, shuffle: bool = True):
        self.pos_indices = [i for i, l in enumerate(labels) if l == 1]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0]
        self.half = batch_size // 2
        self.shuffle = shuffle
        
        logger.info(f"BalancedBatchSampler: {len(self.pos_indices)} positive, "
                    f"{len(self.neg_indices)} negative, half_batch={self.half}")
    
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


# ============================================================================
# SoftPrefixWrapper
# ============================================================================

class SoftPrefixWrapper(nn.Module):
    """
    Wraps HuggingFace model to prepend learnable soft prefix embeddings.
    
    Architecture:
        [SOFT PREFIX (P tokens)] + [INPUT TOKENS (T tokens)]
        
    The prefix is at the embedding level, not text level.
    Final token (Answer:) representations are extracted for probing.
    """
    
    def __init__(self, model, tokenizer, prompt_len: int):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        hidden_dim = model.config.hidden_size
        
        # Freeze model explicitly
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Learnable prefix - small init to avoid dominating early
        self.soft_prefix = nn.Parameter(torch.randn(1, prompt_len, hidden_dim) * 0.01)
        
        logger.info(f"SoftPrefixWrapper initialized: prompt_len={prompt_len}, hidden_dim={hidden_dim}")
    
    def forward(self, texts: List[str]):
        """
        Forward pass with soft prefix prepended.
        
        Returns:
            hidden_states: Tuple of (L+1) tensors
            combined_mask: [B, P+T] attention mask
        """
        # CRITICAL: Left-truncate to preserve Answer: at tail
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(self.model.device)
        
        # Runtime check: skip for raw mode, required for prompted mode
        global RAW_MODE
        if not RAW_MODE:
            for i, ids in enumerate(inputs.input_ids):
                decoded = self.tokenizer.decode(ids[-30:])
                assert "Answer" in decoded or ":" in decoded, \
                    f"Truncation removed Answer: from sample {i}. Decoded tail: {decoded}"
        
        input_ids = inputs.input_ids  # [B, T]
        attention_mask = inputs.attention_mask  # [B, T]
        B, T = input_ids.shape
        
        # Token embeddings
        token_embeds = self.model.model.embed_tokens(input_ids)  # [B, T, D]
        
        # Expand prefix: [1, P, D] -> [B, P, D], align dtype AND device
        prefix_embeds = self.soft_prefix.to(device=token_embeds.device, dtype=token_embeds.dtype)
        prefix_embeds = prefix_embeds.expand(B, -1, -1)
        
        # Concat embeddings: [B, P+T, D]
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        # Concat masks: prefix (all 1s) + original
        P = self.prompt_len
        prefix_mask = torch.ones(B, P, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+T]
        
        # Forward with inputs_embeds
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        
        return outputs.hidden_states, combined_mask
    
    def get_final_token_features(self, texts: List[str], layer_idx: int) -> torch.Tensor:
        """
        Extract hidden state at final token for specified layer.
        
        Args:
            texts: List of model_input strings
            layer_idx: Transformer block index (0-indexed), maps to hidden_states[layer_idx+1]
        
        Returns:
            features: [B, D] tensor
        """
        hidden_states, combined_mask = self.forward(texts)
        
        # layer_idx is 0-indexed transformer block -> hidden_states[layer_idx+1]
        # hidden_states[0] is embeddings, hidden_states[1] is first transformer block
        layer_hidden = hidden_states[layer_idx + 1]  # [B, P+T, D]
        
        # Correctness > speed: extract final token for each sample
        features = []
        for i in range(len(texts)):
            real_len = combined_mask[i].sum().item()
            final_idx = int(real_len) - 1
            features.append(layer_hidden[i, final_idx, :])
        
        return torch.stack(features)  # [B, D]


# ============================================================================
# Probe Loader
# ============================================================================

def load_probe(probe_path: str, input_dim: int, pooling_type: str = "none") -> nn.Module:
    """
    Load trained probe with strict key checking.
    
    Args:
        probe_path: Path to probe checkpoint (.pt file)
        input_dim: Hidden dimension (D), must match probe
        pooling_type: Pooling used by probe (usually 'none' for prompted probes)
    
    Returns:
        Frozen probe module
    """
    from actprobe.probes.models import LayerProbe
    
    probe = LayerProbe(input_dim=input_dim, pooling_type=pooling_type)
    state_dict = torch.load(probe_path, map_location='cpu')
    
    # strict=True ensures all keys match - catches architecture mismatches
    probe.load_state_dict(state_dict, strict=True)
    probe.eval()
    
    # Freeze all parameters
    for param in probe.parameters():
        param.requires_grad = False
    
    logger.info(f"Loaded probe from {probe_path} (input_dim={input_dim}, pooling={pooling_type})")
    return probe


# ============================================================================
# Contrastive Separation Loss
# ============================================================================

def contrastive_separation_loss(
    logits: torch.Tensor,       # [B]
    labels: torch.Tensor,       # [B]
    soft_prefix: nn.Parameter,
    margin: float = 0.5,
    lambda_dist: float = 1e-3,
    lambda_center: float = 1e-2
) -> Tuple[torch.Tensor, dict]:
    """
    Compute contrastive separation loss on logits + regularization.
    
    Loss = max(0, margin - (mu_dec - mu_hon)) + lambda_dist * ||prefix|| + lambda_center * mu_hon^2
    
    The centering term anchors honest logits near 0 to prevent global shift.
    
    Args:
        logits: Probe output logits [B]
        labels: Binary labels [B] (0=honest, 1=deceptive)
        soft_prefix: Learnable prefix parameter
        margin: Minimum desired separation
        lambda_dist: Regularization weight on prefix norm
        lambda_center: Weight on centering term
    
    Returns:
        loss: Scalar tensor
        metrics: Dict with mu_dec, mu_hon, separation, component losses
    """
    # Split by label
    z_dec = logits[labels == 1]  # Deceptive
    z_hon = logits[labels == 0]  # Honest
    
    if len(z_dec) == 0 or len(z_hon) == 0:
        raise ValueError(f"Batch must contain both classes. Got {len(z_dec)} deceptive, {len(z_hon)} honest")
    
    # Mean logits per class
    mu_dec = z_dec.mean()
    mu_hon = z_hon.mean()
    separation = mu_dec - mu_hon
    
    # Main contrastive loss: we want mu_dec > mu_hon by at least margin
    L_contrast = F.relu(margin - separation)
    
    # Regularization on prefix norm (prevent prefix from dominating)
    L_dist = soft_prefix.norm()
    
    # Centering regularizer: anchor honest logits near 0 to prevent global shift
    L_center = mu_hon ** 2
    
    # Total loss
    loss = L_contrast + lambda_dist * L_dist + lambda_center * L_center
    
    metrics = {
        'mu_dec': mu_dec.item(),
        'mu_hon': mu_hon.item(),
        'separation': separation.item(),
        'L_contrast': L_contrast.item(),
        'L_dist': L_dist.item(),
        'L_center': L_center.item(),
        'loss': loss.item()
    }
    
    return loss, metrics


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_prefix_checkpoint(
    prefix: torch.Tensor,
    out_dir: str,
    config: dict
):
    """Save prefix checkpoint with metadata."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save tensor as [P, D] (squeeze batch dim if present)
    prefix_2d = prefix.squeeze(0) if prefix.dim() == 3 else prefix
    save_file({'soft_prefix': prefix_2d}, os.path.join(out_dir, 'prefix.safetensors'))
    
    # Save config
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved prefix checkpoint to {out_dir}")


# ============================================================================
# Training Loop
# ============================================================================

def train_soft_prefix(
    wrapper: SoftPrefixWrapper,
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_idx: int,
    args
) -> torch.Tensor:
    """
    Main training loop for soft prefix.
    
    Returns:
        Best prefix tensor [1, P, D]
    """
    device = next(wrapper.model.parameters()).device
    probe = probe.to(device)
    
    # Only optimize soft_prefix
    optimizer = torch.optim.AdamW([wrapper.soft_prefix], lr=args.lr)
    
    best_separation = -float('inf')
    best_state = None
    
    step = 0
    epoch = 0
    pbar = tqdm(total=args.steps, desc="Training soft prefix")
    
    while step < args.steps:
        epoch += 1
        for texts, labels in train_loader:
            if step >= args.steps:
                break
            
            # Labels already tensor from collate_fn
            labels = labels.to(device)
            
            # Forward: wrapper -> features -> probe
            features = wrapper.get_final_token_features(texts, layer_idx)
            
            # Layer-agnostic probes with pooling (last/mean/max) expect [B, T, D]
            # We have [B, D], so unsqueeze to [B, 1, D] for pooling to work
            # Probes with pooling="none" expect [B, D] directly
            if hasattr(probe, 'pooling') and probe.pooling is not None:
                features = features.unsqueeze(1)  # [B, 1, D]
            
            logits = probe(features).squeeze(-1)  # [B]
            
            # Loss
            loss, metrics = contrastive_separation_loss(
                logits, labels, wrapper.soft_prefix,
                margin=args.margin,
                lambda_dist=args.lambda_dist,
                lambda_center=args.lambda_center
            )
            
            # Backward - only updates soft_prefix
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if step % args.log_every == 0:
                logger.info(
                    f"Step {step:5d} | sep={metrics['separation']:+.4f} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"mu_d={metrics['mu_dec']:+.2f} mu_h={metrics['mu_hon']:+.2f} | "
                    f"L_c={metrics['L_contrast']:.3f} L_d={metrics['L_dist']:.3f}"
                )
            
            # Track best by separation
            if metrics['separation'] > best_separation:
                best_separation = metrics['separation']
                best_state = wrapper.soft_prefix.detach().clone()
            
            pbar.update(1)
            pbar.set_postfix({
                'sep': f"{metrics['separation']:.3f}",
                'loss': f"{metrics['loss']:.3f}"
            })
            step += 1
    
    pbar.close()
    
    # Restore best
    if best_state is not None:
        wrapper.soft_prefix.data = best_state
        logger.info(f"Restored best prefix with separation={best_separation:.4f}")
    
    return wrapper.soft_prefix.detach().cpu()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train soft prefix for prompted probes")
    
    # Model & data
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--train_jsonl", type=str, default=None,
                        help="Path to prepared train JSONL (with suffix)")
    parser.add_argument("--raw_yaml", type=str, default=None,
                        help="Path to raw YAML data (without suffix, e.g., roleplaying/dataset.yaml)")
    parser.add_argument("--val_jsonl", type=str, default=None,
                        help="Path to prepared val JSONL (optional)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models")
    
    # Probe
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to probe checkpoint (.pt)")
    parser.add_argument("--layer_idx", type=int, required=True,
                        help="Layer index for probing (0-indexed)")
    parser.add_argument("--pooling", type=str, default="none",
                        help="Pooling type used by probe")
    
    # Prefix hyperparams
    parser.add_argument("--prompt_len", type=int, default=16,
                        help="Soft prefix length (number of virtual tokens)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Contrastive margin")
    parser.add_argument("--lambda_dist", type=float, default=1e-3,
                        help="Regularization weight on prefix norm")
    parser.add_argument("--lambda_center", type=float, default=1e-2,
                        help="Weight on centering term")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (will be split B/2 pos, B/2 neg)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N steps")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="checkpoints/soft_prefix",
                        help="Output directory for checkpoint")
    parser.add_argument("--probe_type", type=str, default="single_layer",
                        help="Probe type for directory naming")
    
    args = parser.parse_args()
    
    # ========================================================================
    # Setup
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("Soft Prefix Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Probe: {args.probe_path} (layer {args.layer_idx})")
    logger.info(f"Prefix length: {args.prompt_len}")
    logger.info(f"Margin: {args.margin}, lambda_dist: {args.lambda_dist}, lambda_center: {args.lambda_center}")
    logger.info("=" * 70)
    
    # ========================================================================
    # Load Model
    # ========================================================================
    
    logger.info("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=args.hf_token,
        padding_side="left"  # For decoder-only models
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    logger.info(f"Model loaded on {device}, hidden_dim={hidden_dim}")
    
    # ========================================================================
    # Load Probe
    # ========================================================================
    
    probe = load_probe(args.probe_path, hidden_dim, args.pooling)
    
    # ========================================================================
    # Create Wrapper
    # ========================================================================
    
    wrapper = SoftPrefixWrapper(model, tokenizer, args.prompt_len)
    
    # ========================================================================
    # Load Datasets
    # ========================================================================
    
    global RAW_MODE
    
    if args.raw_yaml:
        # Raw mode: no suffix, just scenario + completion
        RAW_MODE = True
        logger.info(f"Using RAW mode with YAML data (no suffix)")
        train_dataset = RawDataset(args.raw_yaml)
        val_dataset = train_dataset  # Use same for validation in raw mode
    elif args.train_jsonl:
        # Prompted mode: with suffix format
        RAW_MODE = False
        train_dataset = PreparedDataset(args.train_jsonl)
        val_dataset = PreparedDataset(args.val_jsonl) if args.val_jsonl else train_dataset
    else:
        raise ValueError("Must provide either --train_jsonl or --raw_yaml")
    
    # Get labels for balanced sampler
    train_labels = train_dataset.get_labels()
    
    # Create balanced sampler
    batch_sampler = BalancedBatchSampler(train_labels, args.batch_size)
    
    # NOTE: Use batch_sampler=, NOT sampler=
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # ========================================================================
    # Train
    # ========================================================================
    
    prefix = train_soft_prefix(wrapper, probe, train_loader, val_loader, args.layer_idx, args)
    
    # ========================================================================
    # Save Checkpoint
    # ========================================================================
    
    # Build output directory
    model_name = args.model.replace("/", "_")
    out_subdir = os.path.join(
        args.out_dir,
        model_name,
        f"{args.probe_type}_{args.pooling}_layer{args.layer_idx}"
    )
    
    config = {
        "model": args.model,
        "probe_path": args.probe_path,
        "layer_idx": args.layer_idx,
        "pooling": args.pooling,
        "prompt_len": args.prompt_len,
        "margin": args.margin,
        "lambda_dist": args.lambda_dist,
        "lambda_center": args.lambda_center,
        "lr": args.lr,
        "steps": args.steps,
        "train_jsonl": args.train_jsonl
    }
    
    save_prefix_checkpoint(prefix, out_subdir, config)
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Checkpoint saved to: {out_subdir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
