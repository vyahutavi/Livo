"""
LIVO Audio Encoder — From-scratch audio understanding.

Converts raw audio (mel spectrograms) into embeddings that LIVO's
transformer decoder can process. Uses Conv1D for feature extraction
and shared TransformerBlocks for sequence modeling.

Architecture:
    Mel Spectrogram → Conv1D stack → TransformerBlocks → Audio Embeddings (d_model)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


@dataclass
class AudioEncoderConfig:
    """Configuration for the audio encoder."""
    n_mels: int = 80            # Number of mel frequency bins
    d_model: int = 768          # Must match LIVO's d_model
    num_layers: int = 3         # Transformer layers for audio
    num_heads: int = 12         # Attention heads
    ffn_dim: int = 3072         # FFN hidden dimension
    dropout: float = 0.1
    conv_channels: list = None  # Conv1D channel progression

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [128, 256, self.d_model]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioEncoderConfig":
        cfg = data.get("audio_encoder", data)
        return cls(
            n_mels=cfg.get("n_mels", cls.n_mels),
            d_model=cfg.get("d_model", cls.d_model),
            num_layers=cfg.get("num_layers", cls.num_layers),
            num_heads=cfg.get("num_heads", cls.num_heads),
            ffn_dim=cfg.get("ffn_dim", cls.ffn_dim),
            dropout=cfg.get("dropout", cls.dropout),
            conv_channels=cfg.get("conv_channels", None),
        )


class AudioEncoder(nn.Module):
    """
    From-scratch audio encoder for LIVO.

    Converts mel spectrograms into a sequence of embeddings that can be
    concatenated with text embeddings and fed into LIVO's transformer decoder.

    Pipeline:
        1. Conv1D stack: Extract local audio features + downsample
        2. Positional encoding: Add learned position information
        3. Transformer layers: Model long-range audio dependencies
        4. Output: (batch, audio_seq_len, d_model) — same dim as text embeddings

    Args:
        config: AudioEncoderConfig with architecture parameters

    Example:
        encoder = AudioEncoder(AudioEncoderConfig(d_model=384))
        mel = torch.randn(2, 80, 300)  # (batch, n_mels, time_frames)
        audio_embeds = encoder(mel)     # (2, 75, 384)
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # --- Conv1D Feature Extraction ---
        # Each conv layer extracts increasingly abstract audio features.
        # Stride=2 on layers 2+ downsamples the time axis (reduces sequence length).
        channels = config.conv_channels
        conv_layers = []
        in_channels = config.n_mels

        for i, out_channels in enumerate(channels):
            stride = 1 if i == 0 else 2
            conv_layers.append(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1
                )
            )
            conv_layers.append(nn.GELU())
            conv_layers.append(nn.Dropout(config.dropout))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*conv_layers)

        # --- Positional Encoding ---
        # Maximum audio sequence length after downsampling.
        # For 30s audio at 16kHz with hop=160: ~3000 frames → ~750 after 2x downsample
        self.max_audio_positions = 2048
        self.position_embedding = nn.Embedding(self.max_audio_positions, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # --- Transformer Layers ---
        # Reuse the SAME TransformerBlock from LIVO's text model.
        # Audio uses FULL attention (not causal) because we see the whole audio.
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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize conv layers with standard normal distribution."""
        for module in self.conv_stack.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a mel spectrogram into LIVO-compatible embeddings.

        Args:
            mel_spectrogram: (batch, n_mels, time_frames)
                Mel spectrogram from torchaudio or librosa.
            attention_mask: Optional (batch, audio_seq_len) boolean mask.

        Returns:
            Audio embeddings of shape (batch, audio_seq_len, d_model).
            These can be concatenated with text embeddings for multimodal input.
        """
        # 1. Conv1D feature extraction: (batch, n_mels, T) → (batch, d_model, T')
        x = self.conv_stack(mel_spectrogram)

        # 2. Transpose to sequence format: (batch, T', d_model)
        x = x.transpose(1, 2)

        batch_size, seq_len, _ = x.shape

        # 3. Add positional encoding
        if seq_len > self.max_audio_positions:
            raise ValueError(
                f"Audio sequence length {seq_len} exceeds maximum "
                f"{self.max_audio_positions}. Use shorter audio clips."
            )

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)

        # 4. Transformer layers (FULL attention — audio is not causal)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask, causal=False)

        # 5. Final normalization
        x = self.output_norm(x)

        return x  # (batch, audio_seq_len, d_model)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
