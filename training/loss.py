"""Loss functions for causal language modeling."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for causal language modeling.

    Shifts logits and labels so the model predicts the next token at each position.

    Args:
        logits: Model output of shape (batch, seq_len, vocab_size)
        labels: Target token IDs of shape (batch, seq_len)
        ignore_index: Token ID to ignore in loss (typically pad_token_id)

    Returns:
        Scalar loss tensor
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


def perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss to perplexity with overflow protection."""
    try:
        return math.exp(min(loss, 100.0))
    except OverflowError:
        return float("inf")
