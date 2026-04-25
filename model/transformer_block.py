from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _attn_ffn_block(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attention + FFN block (wrapped for gradient checkpointing)."""
        normed_states = self.attn_norm(hidden_states)
        attn_output, _ = self.attention(
            normed_states,
            normed_states,
            normed_states,
            attn_mask=attn_mask,
            key_padding_mask=~attention_mask if attention_mask is not None else None,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attn_dropout(attn_output)

        ffn_output = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + self.ffn_dropout(ffn_output)

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape (batch, seq, dim), "
                f"got {tuple(hidden_states.shape)}"
            )

        batch_size, seq_len, _ = hidden_states.shape
        del batch_size

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)

        attn_mask = None
        if causal:
            attn_mask = self._causal_mask(seq_len, hidden_states.device)

        if self.use_gradient_checkpointing and self.training:
            hidden_states = checkpoint(
                self._attn_ffn_block,
                hidden_states,
                attention_mask,
                attn_mask,
                use_reentrant=False,
            )
        else:
            hidden_states = self._attn_ffn_block(hidden_states, attention_mask, attn_mask)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        return hidden_states
