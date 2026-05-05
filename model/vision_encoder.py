"""
LIVO Vision Encoder — From-scratch image understanding.

Converts images into patch embeddings that LIVO's transformer decoder
can process. Uses the Vision Transformer (ViT) approach: split image
into fixed-size patches, embed each patch, and process with TransformerBlocks.

Architecture:
    Image → Patch Split → Linear Projection → TransformerBlocks → Vision Embeddings (d_model)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder."""
    img_size: int = 224         # Input image size (square)
    patch_size: int = 16        # Size of each patch (16x16 pixels)
    in_channels: int = 3        # RGB channels
    d_model: int = 768          # Must match LIVO's d_model
    num_layers: int = 4         # Transformer layers for vision
    num_heads: int = 12         # Attention heads
    ffn_dim: int = 3072         # FFN hidden dimension
    dropout: float = 0.1
    use_cls_token: bool = True  # Add a [CLS] token for global image representation

    @property
    def num_patches(self) -> int:
        """Number of patches the image is split into."""
        return (self.img_size // self.patch_size) ** 2  # 196 for 224/16

    @property
    def patch_dim(self) -> int:
        """Flattened dimension of a single patch."""
        return self.in_channels * self.patch_size * self.patch_size  # 768 for 3*16*16

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionEncoderConfig":
        cfg = data.get("vision_encoder", data)
        return cls(
            img_size=cfg.get("img_size", cls.img_size),
            patch_size=cfg.get("patch_size", cls.patch_size),
            in_channels=cfg.get("in_channels", cls.in_channels),
            d_model=cfg.get("d_model", cls.d_model),
            num_layers=cfg.get("num_layers", cls.num_layers),
            num_heads=cfg.get("num_heads", cls.num_heads),
            ffn_dim=cfg.get("ffn_dim", cls.ffn_dim),
            dropout=cfg.get("dropout", cls.dropout),
            use_cls_token=cfg.get("use_cls_token", cls.use_cls_token),
        )


class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and project each to d_model.

    This is the "tokenizer" for images — just like livorator splits text into
    subword tokens, PatchEmbedding splits images into visual tokens.

    Input:  (batch, 3, 224, 224)
    Output: (batch, 196, d_model)
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches

        # Conv2d with kernel_size=stride=patch_size acts as a patch extractor
        # This is mathematically equivalent to reshape + linear, but faster.
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (batch, channels, height, width)

        Returns:
            Patch embeddings: (batch, num_patches, d_model)
        """
        # (batch, d_model, grid_h, grid_w)
        x = self.projection(image)
        # Flatten spatial dims: (batch, d_model, num_patches) → (batch, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionEncoder(nn.Module):
    """
    From-scratch Vision Transformer encoder for LIVO.

    Converts images into a sequence of embeddings that can be concatenated
    with text embeddings and fed into LIVO's transformer decoder.

    This follows the ViT (Vision Transformer) architecture:
    1. Split image into 16×16 patches (like visual "words")
    2. Project each patch to d_model dimensions
    3. Add positional embeddings (so the model knows patch locations)
    4. Process through transformer layers
    5. Output embeddings compatible with LIVO's text embeddings

    Args:
        config: VisionEncoderConfig with architecture parameters

    Example:
        encoder = VisionEncoder(VisionEncoderConfig(d_model=384))
        image = torch.randn(2, 3, 224, 224)   # (batch, RGB, H, W)
        vision_embeds = encoder(image)          # (2, 197, 384) with CLS
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        # --- Patch Embedding (the "visual tokenizer") ---
        self.patch_embedding = PatchEmbedding(config)

        # --- CLS Token ---
        # A special learnable token prepended to the sequence.
        # After processing, it contains a global summary of the entire image.
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # --- Position Embedding ---
        num_positions = config.num_patches + (1 if config.use_cls_token else 0)
        self.position_embedding = nn.Embedding(num_positions, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # --- Transformer Layers ---
        # Reuse the SAME TransformerBlock from LIVO's text model.
        # Vision uses FULL attention (not causal) because we see the whole image.
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_gradient_checkpointing=True,
            )
            for _ in range(config.num_layers)
        ])

        self.output_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projection weights."""
        for module in self.patch_embedding.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into LIVO-compatible embeddings.

        Args:
            image: (batch, channels, height, width)
                Normalized image tensor. Expected: (B, 3, 224, 224)

        Returns:
            Vision embeddings of shape (batch, num_patches + 1, d_model).
            The +1 is the CLS token (if enabled).
            These can be concatenated with text embeddings for multimodal input.
        """
        batch_size = image.shape[0]

        # 1. Split into patches and embed: (batch, num_patches, d_model)
        x = self.patch_embedding(image)

        # 2. Prepend CLS token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, d_model)

        # 3. Add positional encoding
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # 4. Transformer layers (FULL attention — vision is not causal)
        for layer in self.transformer_layers:
            x = layer(x, causal=False)

        # 5. Final normalization
        x = self.output_norm(x)

        return x  # (batch, num_patches + 1, d_model)

    def get_cls_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get only the CLS token embedding (global image representation).

        Useful for image classification tasks.

        Args:
            image: (batch, channels, height, width)

        Returns:
            CLS embedding: (batch, d_model)
        """
        if not self.config.use_cls_token:
            raise ValueError("CLS token is disabled in config.")
        embeddings = self.forward(image)
        return embeddings[:, 0, :]  # First token is CLS

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
