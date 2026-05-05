"""
LIVO Multimodal — Combines text, audio, vision, and speech into one model.

This is the glue that connects LIVO's language decoder with the audio
and vision encoders, plus the speech decoder for voice output.

Architecture:
    INPUTS                          BRAIN                         OUTPUTS
    ┌─────────────┐                                            ┌───────────┐
    │ Vision Enc.  │──┐                                        │  Text     │
    └─────────────┘  │    ┌─────────────────────┐             │ (logits)  │
                     ├──► │   LIVO Transformer   │──► logits──►└───────────┘
    ┌─────────────┐  │    │   (Shared Decoder)   │
    │ Audio Enc.   │──┤    │  12 layers × 768d   │──► embeds──►┌───────────┐
    └─────────────┘  │    └─────────────────────┘             │  Speech   │
                     │                                         │ Decoder   │
    ┌─────────────┐  │                                         │ + Vocoder │
    │ livorator    │──┘                                        └───────────┘
    └─────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.llm import LLM, Config, CausalLMOutput
from model.audio_encoder import AudioEncoder, AudioEncoderConfig
from model.vision_encoder import VisionEncoder, VisionEncoderConfig
from model.speech_decoder import SpeechDecoder, SpeechDecoderConfig


@dataclass
class MultimodalConfig:
    """Configuration for the full multimodal LIVO model."""
    # LLM (text) config
    llm: Config = None
    # Encoder configs (None = disabled)
    audio: Optional[AudioEncoderConfig] = None
    vision: Optional[VisionEncoderConfig] = None
    # Decoder configs (None = disabled)
    speech: Optional[SpeechDecoderConfig] = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = Config()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalConfig":
        llm_config = Config.from_dict(data)

        audio_config = None
        if "audio_encoder" in data:
            audio_cfg = data["audio_encoder"]
            audio_cfg.setdefault("d_model", llm_config.d_model)
            audio_config = AudioEncoderConfig.from_dict(audio_cfg)

        vision_config = None
        if "vision_encoder" in data:
            vision_cfg = data["vision_encoder"]
            vision_cfg.setdefault("d_model", llm_config.d_model)
            vision_config = VisionEncoderConfig.from_dict(vision_cfg)

        speech_config = None
        if "speech_decoder" in data:
            speech_cfg = data["speech_decoder"]
            speech_cfg.setdefault("d_model", llm_config.d_model)
            speech_config = SpeechDecoderConfig.from_dict(speech_cfg)

        return cls(
            llm=llm_config,
            audio=audio_config,
            vision=vision_config,
            speech=speech_config,
        )


class MultimodalLIVO(nn.Module):
    """
    Multimodal LIVO — A from-scratch multimodal language model.

    Combines vision (images), audio (speech/sound), and text into a
    single unified model with optional speech output. Each modality
    has its own encoder, but they all share the same transformer
    decoder (the LIVO brain).

    The key insight: all modalities are converted to the same d_model
    dimensional embeddings, then concatenated into one sequence:

        [vision_patches, audio_frames, text_tokens]

    The transformer processes this combined sequence and generates
    text output conditioned on all input modalities. Additionally,
    the hidden states can be routed through the speech decoder to
    produce audio output.

    Args:
        config: MultimodalConfig specifying all sub-model architectures

    Example:
        config = MultimodalConfig(
            llm=Config(d_model=384),
            vision=VisionEncoderConfig(d_model=384),
            audio=AudioEncoderConfig(d_model=384),
            speech=SpeechDecoderConfig(d_model=384),
        )
        model = MultimodalLIVO(config)

        # Text → Text + Speech
        out = model(text_ids=token_ids)
        speech = model.speak(out.hidden_states)
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # --- Text Model (always present) ---
        self.llm = LLM(config.llm)

        # --- Vision Encoder (optional) ---
        self.vision_encoder = None
        if config.vision is not None:
            assert config.vision.d_model == config.llm.d_model, (
                f"Vision d_model ({config.vision.d_model}) must match "
                f"LLM d_model ({config.llm.d_model})"
            )
            self.vision_encoder = VisionEncoder(config.vision)

        # --- Audio Encoder (optional) ---
        self.audio_encoder = None
        if config.audio is not None:
            assert config.audio.d_model == config.llm.d_model, (
                f"Audio d_model ({config.audio.d_model}) must match "
                f"LLM d_model ({config.llm.d_model})"
            )
            self.audio_encoder = AudioEncoder(config.audio)

        # --- Speech Decoder (optional) ---
        self.speech_decoder = None
        if config.speech is not None:
            assert config.speech.d_model == config.llm.d_model, (
                f"Speech d_model ({config.speech.d_model}) must match "
                f"LLM d_model ({config.llm.d_model})"
            )
            self.speech_decoder = SpeechDecoder(config.speech)

        # --- Modality Type Embeddings ---
        # Small learned vectors that tell the model "this is vision"
        # vs "this is audio" vs "this is text" — like segment embeddings in BERT.
        num_modalities = 3  # text=0, vision=1, audio=2
        self.modality_embedding = nn.Embedding(num_modalities, config.llm.d_model)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)

    def _get_text_embeddings(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from the LLM's embedding layers."""
        token_emb = self.llm.token_embedding(text_ids)
        pos_emb = self.llm.position_embedding(text_ids)
        text_emb = token_emb + pos_emb

        # Add modality type = 0 (text)
        modality_id = torch.zeros(1, dtype=torch.long, device=text_ids.device)
        text_emb = text_emb + self.modality_embedding(modality_id)
        return text_emb

    def _get_vision_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image and add vision modality embedding."""
        vision_emb = self.vision_encoder(image)

        # Add modality type = 1 (vision)
        modality_id = torch.ones(1, dtype=torch.long, device=image.device)
        vision_emb = vision_emb + self.modality_embedding(modality_id)
        return vision_emb

    def _get_audio_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio and add audio modality embedding."""
        audio_emb = self.audio_encoder(audio)

        # Add modality type = 2 (audio)
        modality_id = torch.full((1,), 2, dtype=torch.long, device=audio.device)
        audio_emb = audio_emb + self.modality_embedding(modality_id)
        return audio_emb

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        """
        Forward pass through the multimodal model.

        Args:
            text_ids: (batch, text_seq_len) — Token IDs from livorator
            attention_mask: (batch, text_seq_len) — Padding mask for text
            labels: (batch, text_seq_len) — Target token IDs for loss
            image: (batch, 3, H, W) — Input image tensor (optional)
            audio: (batch, n_mels, time_frames) — Mel spectrogram (optional)

        Returns:
            CausalLMOutput with logits, optional loss, and hidden_states.
        """
        if text_ids is None and image is None and audio is None:
            raise ValueError("At least one input modality must be provided.")

        embedding_parts = []
        prefix_length = 0

        # --- Vision embeddings (prepended before text) ---
        if image is not None and self.vision_encoder is not None:
            vision_emb = self._get_vision_embeddings(image)
            embedding_parts.append(vision_emb)
            prefix_length += vision_emb.shape[1]

        # --- Audio embeddings (prepended before text) ---
        if audio is not None and self.audio_encoder is not None:
            audio_emb = self._get_audio_embeddings(audio)
            embedding_parts.append(audio_emb)
            prefix_length += audio_emb.shape[1]

        # --- Text embeddings ---
        if text_ids is not None:
            text_emb = self._get_text_embeddings(text_ids)
            embedding_parts.append(text_emb)

        if not embedding_parts:
            raise ValueError("No valid embeddings produced from inputs.")

        combined = torch.cat(embedding_parts, dim=1)

        # --- Build combined attention mask ---
        if attention_mask is not None and prefix_length > 0:
            batch_size = combined.shape[0]
            prefix_mask = torch.ones(
                batch_size, prefix_length,
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        elif attention_mask is not None:
            combined_mask = attention_mask
        else:
            combined_mask = None

        # --- Pass through LIVO's transformer decoder ---
        x = combined
        for block in self.llm.transformer:
            x = block(x, attention_mask=combined_mask)

        x = self.llm.final_norm(x)
        logits = self.llm.lm_head(x)

        # --- Compute loss (only on text portion) ---
        loss = None
        if labels is not None and text_ids is not None:
            text_logits = logits[:, prefix_length:, :]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.llm.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.llm.pad_token_id,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=x,  # Always return for speech decoder
        )

    def speak(
        self,
        hidden_states: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        max_audio_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert model hidden states to speech (mel spectrogram).

        Args:
            hidden_states: (batch, seq_len, d_model) from forward() output
            target_durations: Ground truth durations (training only)
            max_audio_length: Cap on output mel frames

        Returns:
            Dict with mel, mel_postnet, and durations.
            Use GriffinLimVocoder to convert mel_postnet to audio.

        Example:
            out = model(text_ids=tokens)
            speech = model.speak(out.hidden_states)
            vocoder = GriffinLimVocoder()
            audio = vocoder.synthesize(speech["mel_postnet"])
            vocoder.save_wav(audio, "output.wav")
        """
        if self.speech_decoder is None:
            raise RuntimeError(
                "Speech decoder not configured. Initialize MultimodalConfig "
                "with speech=SpeechDecoderConfig(d_model=384)"
            )
        return self.speech_decoder(
            hidden_states,
            target_durations=target_durations,
            max_audio_length=max_audio_length,
        )

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Parameter count breakdown by component."""
        counts = {
            "llm": sum(p.numel() for p in self.llm.parameters()),
            "modality_embedding": self.modality_embedding.weight.numel(),
        }
        if self.vision_encoder is not None:
            counts["vision_encoder"] = self.vision_encoder.num_parameters
        if self.audio_encoder is not None:
            counts["audio_encoder"] = self.audio_encoder.num_parameters
        if self.speech_decoder is not None:
            counts["speech_decoder"] = self.speech_decoder.num_parameters
        counts["total"] = sum(counts.values())
        return counts

    def freeze_llm(self) -> None:
        """Freeze the LLM weights (for training encoders only)."""
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self) -> None:
        """Unfreeze the LLM weights (for joint fine-tuning)."""
        for param in self.llm.parameters():
            param.requires_grad = True

    def freeze_encoders(self) -> None:
        """Freeze all encoder weights (for training speech decoder only)."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if self.audio_encoder is not None:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze all encoder weights."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
        if self.audio_encoder is not None:
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
