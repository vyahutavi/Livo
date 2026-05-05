from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import LearnedPositionEmbedding, TokenEmbedding
from model.transformer_block import TransformerBlock

@dataclass
class Config:
    vocab_size: int = 50000
    max_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        model_cfg = data.get("model", data)
        return cls(
            vocab_size=model_cfg.get("vocab_size", cls.vocab_size),
            max_length=model_cfg.get("max_length", cls.max_length),
            d_model=model_cfg.get("d_model", cls.d_model),
            num_layers=model_cfg.get("num_layers", cls.num_layers),
            num_heads=model_cfg.get("num_heads", cls.num_heads),
            ffn_dim=model_cfg.get("ffn_dim", cls.ffn_dim),
            dropout=model_cfg.get("dropout", cls.dropout),
        )

@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None

class LLM(nn.Module):
    def __init__(self, config: Config): # Fixed double underscores
        super().__init__()
        self.config = config
        
        # 1. Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            pad_token_id=config.pad_token_id,
        )
        self.position_embedding = LearnedPositionEmbedding(
            max_length=config.max_length,
            d_model=config.d_model,
        )

        # 2. Transformer Blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            ) for _ in range(config.num_layers)
        ])

        # 3. Output Head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._reset_parameters()

        # Weight Tying: Shared weights between input and output
        # Must come AFTER _reset_parameters to preserve the tie
        self.lm_head.weight = self.token_embedding.embedding.weight

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> CausalLMOutput:
        
        # Prepare hidden states
        x = self.token_embedding(input_ids) + self.position_embedding(input_ids)

        # Process through transformer layers
        for block in self.transformer:
            # Traditional Transformers use causal mask internally in TransformerBlock
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift for Causal LM training
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=x if return_hidden_states else None
        )