"""
Token pooling strategies for reducing (T, D) -> (D).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, D)
        """
        raise NotImplementedError

class MeanPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> mean(dim=1) -> (B, D)
        return x.mean(dim=1)

class MaxPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> max(dim=1) -> (B, D)
        return x.max(dim=1)[0]

class LastTokenPooling(BasePooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> last index -> (B, D)
        return x[:, -1, :]

class LearnedAttentionPooling(BasePooling):
    """
    Learns a query vector q to compute attention scores over tokens.
    a_t = softmax(x_t * q)
    z = sum(a_t * x_t)

    Optionally returns attention weights for entropy analysis (Spec section 9A).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim, 1))
        self.last_attention_weights = None  # Store for analysis

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, D)
            return_attention: If True, returns (pooled, attention_weights)

        Returns:
            pooled: (B, D) or (pooled, weights) if return_attention=True
        """
        # x: (B, T, D)
        # scores = x @ q -> (B, T, 1)
        scores = torch.matmul(x, self.query)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)

        # Store for analysis (spec: log entropy of attention weights)
        self.last_attention_weights = weights.detach()

        # Weighted sum: (B, T, D) * (B, T, 1) -> sum(dim=1) -> (B, D)
        pooled = (x * weights).sum(dim=1)

        if return_attention:
            return pooled, weights.squeeze(-1)  # (B, D), (B, T)
        return pooled

    def compute_attention_entropy(self):
        """
        Compute Shannon entropy of last attention weights.
        Returns mean entropy across batch.

        H(weights) = -sum(w * log(w))
        Lower entropy = more concentrated attention (spec requirement).
        """
        if self.last_attention_weights is None:
            return None

        weights = self.last_attention_weights.squeeze(-1)  # (B, T)
        # Avoid log(0)
        weights = torch.clamp(weights, min=1e-10)
        entropy = -(weights * torch.log(weights)).sum(dim=1)  # (B,)
        return entropy.mean().item()


class RMSNorm1D(nn.Module):
    """RMSNorm for the last dimension of a tensor."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class GMHAAttentionPooling(BasePooling):
    """
    Gated Multi-Head Attention-style pooling with one learned query per head.

    This follows the probe-side structure used in the paper excerpt:
      - learned query per head
      - K,V projections from token embeddings
      - masked softmax over tokens
      - concatenate head outputs, then output projection
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        output_dim: Optional[int] = None,
        topk_tokens: Optional[int] = None,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.input_dim = input_dim
        self.num_heads = num_heads
        # Allow arbitrary head counts by projecting into the nearest valid head space.
        self.head_dim = int(math.ceil(float(input_dim) / float(num_heads)))
        self.proj_dim = self.num_heads * self.head_dim
        self.output_dim = output_dim or input_dim
        self.topk_tokens = topk_tokens

        self.w_kv = nn.Linear(input_dim, 2 * self.proj_dim, bias=False)
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.w_out = nn.Linear(self.proj_dim, self.output_dim)
        self.last_attention_weights = None

    def _compute_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        bsz, seq_len, _ = x.shape
        kv = self.w_kv(x).view(bsz, seq_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, T, Dh)
        k, v = kv[0], kv[1]

        # (B, H, T, Dh) dot (H, Dh) -> (B, H, T)
        logits = torch.einsum("bhtd,hd->bht", k, self.query)
        logits = logits / (float(self.head_dim) ** 0.5)
        return logits, v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # x: (B, T, D)
        logits, v = self._compute_scores(x)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            logits = logits.masked_fill(~attention_mask[:, None, :], float("-inf"))

        if self.topk_tokens is not None and self.topk_tokens > 0 and self.topk_tokens < logits.shape[-1]:
            keep_idx = torch.topk(logits, k=self.topk_tokens, dim=-1).indices
            topk_mask = torch.zeros_like(logits, dtype=torch.bool)
            topk_mask.scatter_(-1, keep_idx, True)
            logits = logits.masked_fill(~topk_mask, float("-inf"))

        scores = F.softmax(logits.float(), dim=-1).to(v.dtype)  # (B, H, T)
        self.last_attention_weights = scores.detach()

        # (B, H, T) x (B, H, T, Dh) -> (B, H, Dh)
        pooled = torch.einsum("bht,bhtd->bhd", scores, v)
        pooled = pooled.reshape(x.shape[0], self.num_heads * self.head_dim)  # (B, D)
        out = self.w_out(pooled)

        if return_attention:
            return out, scores
        return out


class MultiMaxPooling(GMHAAttentionPooling):
    """Multi-head attention pooling with top-k token selection per head."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        output_dim: Optional[int] = None,
        topk_tokens: int = 4,
    ):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            topk_tokens=topk_tokens,
        )


class RollingAttentionPooling(BasePooling):
    """
    Rolling window variant:
      1) Apply GMHA pooling per sliding window.
      2) Mean-pool pooled window vectors into a final vector.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        output_dim: Optional[int] = None,
        window_size: int = 64,
        stride: int = 32,
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.input_dim = input_dim
        self.window_size = window_size
        self.stride = stride
        self.inner = GMHAAttentionPooling(
            input_dim=input_dim,
            num_heads=num_heads,
            output_dim=output_dim or input_dim,
        )
        self.last_attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # x: (B, T, D)
        bsz, seq_len, _ = x.shape
        window_outputs = []
        attn_per_window = []

        starts = list(range(0, max(1, seq_len), self.stride))
        if starts and starts[-1] + self.window_size < seq_len:
            starts.append(seq_len - self.window_size)
        starts = sorted(set(max(0, s) for s in starts))

        for start in starts:
            end = min(seq_len, start + self.window_size)
            x_win = x[:, start:end, :]
            if x_win.shape[1] == 0:
                continue

            if attention_mask is not None:
                m_win = attention_mask[:, start:end]
            else:
                m_win = None

            pooled, attn = self.inner(x_win, attention_mask=m_win, return_attention=True)
            window_outputs.append(pooled)
            attn_per_window.append(attn)

        if not window_outputs:
            # Should never happen with sane inputs, but keep a robust fallback.
            out = self.inner(x, attention_mask=attention_mask)
            if return_attention:
                return out, None
            return out

        stacked = torch.stack(window_outputs, dim=1)  # (B, W, D)
        out = stacked.mean(dim=1)  # (B, D)
        self.last_attention_weights = attn_per_window

        if return_attention:
            return out, attn_per_window
        return out
