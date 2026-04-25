from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding table with padding awareness."""

    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LearnedPositionEmbedding(nn.Module):
    """Learned positional embeddings for autoregressive transformer blocks."""

    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_length {self.max_length}"
            )

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        return self.embedding(position_ids)