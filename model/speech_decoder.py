"""
LIVO Speech Decoder — From-scratch text-to-speech output.

Converts LIVO's text embeddings into mel spectrograms, which are then
converted to audio waveforms using the Griffin-Lim vocoder.

Architecture:
    LIVO Text Embeddings → TransformerBlocks → Linear → Mel Spectrogram
    Mel Spectrogram → Griffin-Lim Vocoder → Audio Waveform (.wav)

This gives LIVO a "voice" — the model can speak its outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


@dataclass
class SpeechDecoderConfig:
    """Configuration for the speech decoder."""
    d_model: int = 768          # Must match LIVO's d_model
    n_mels: int = 80            # Mel spectrogram frequency bins
    num_layers: int = 3         # Transformer layers for speech
    num_heads: int = 12         # Attention heads
    ffn_dim: int = 3072         # FFN hidden dimension
    dropout: float = 0.1
    max_audio_length: int = 1024  # Max output mel frames
    num_postnet_layers: int = 3   # Post-processing conv layers
    postnet_channels: int = 256   # Channels in PostNet

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeechDecoderConfig":
        cfg = data.get("speech_decoder", data)
        return cls(
            d_model=cfg.get("d_model", cls.d_model),
            n_mels=cfg.get("n_mels", cls.n_mels),
            num_layers=cfg.get("num_layers", cls.num_layers),
            num_heads=cfg.get("num_heads", cls.num_heads),
            ffn_dim=cfg.get("ffn_dim", cls.ffn_dim),
            dropout=cfg.get("dropout", cls.dropout),
            max_audio_length=cfg.get("max_audio_length", cls.max_audio_length),
            num_postnet_layers=cfg.get("num_postnet_layers", cls.num_postnet_layers),
            postnet_channels=cfg.get("postnet_channels", cls.postnet_channels),
        )


class PostNet(nn.Module):
    """
    Post-processing network for mel spectrogram refinement.

    Applies a stack of Conv1D layers to smooth and refine the predicted
    mel spectrogram. This significantly improves audio quality.

    Architecture: Conv1D → BatchNorm → Tanh → ... → Conv1D
    """

    def __init__(self, n_mels: int, num_layers: int = 3, channels: int = 256):
        super().__init__()
        layers = []

        # First layer: n_mels → channels
        layers.extend([
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
        ])

        # Middle layers: channels → channels
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
            ])

        # Final layer: channels → n_mels (back to mel dimension)
        layers.extend([
            nn.Conv1d(channels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, time, n_mels)
        Returns:
            Refined mel: (batch, time, n_mels)
        """
        # Conv1D expects (batch, channels, time)
        residual = self.layers(mel.transpose(1, 2)).transpose(1, 2)
        return mel + residual  # Residual connection


class DurationPredictor(nn.Module):
    """
    Predicts how many mel frames each text token should produce.

    This solves the length mismatch problem: text has N tokens,
    but audio needs M frames (usually M >> N). The duration predictor
    learns how long each token should be spoken.

    Architecture: Linear → ReLU → Linear → Softplus (ensures positive durations)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Ensures positive durations
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, text_seq_len, d_model)
        Returns:
            Predicted durations: (batch, text_seq_len) — frames per token
        """
        return self.layers(hidden_states).squeeze(-1)


class LengthRegulator(nn.Module):
    """
    Expands text embeddings to match audio length using predicted durations.

    If token "Hello" has duration 5, its embedding is repeated 5 times.
    This converts (batch, text_len, d_model) → (batch, audio_len, d_model).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        durations: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, text_seq_len, d_model)
            durations: (batch, text_seq_len) — integer frames per token
            max_length: Optional cap on output length

        Returns:
            Expanded embeddings: (batch, audio_seq_len, d_model)
        """
        # Round durations to integers (minimum 1 frame per token)
        durations = torch.clamp(durations.round().long(), min=1)

        batch_size = hidden_states.shape[0]
        expanded = []

        for b in range(batch_size):
            # Repeat each token embedding by its duration
            seq = torch.repeat_interleave(
                hidden_states[b], durations[b], dim=0
            )
            if max_length is not None:
                seq = seq[:max_length]
            expanded.append(seq)

        # Pad to same length within batch
        max_len = max(s.shape[0] for s in expanded)
        if max_length is not None:
            max_len = min(max_len, max_length)

        padded = torch.zeros(
            batch_size, max_len, hidden_states.shape[-1],
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        for b, seq in enumerate(expanded):
            length = min(seq.shape[0], max_len)
            padded[b, :length] = seq[:length]

        return padded


class SpeechDecoder(nn.Module):
    """
    From-scratch speech decoder for LIVO.

    Converts text embeddings from LIVO's transformer into mel spectrograms.
    Uses a duration predictor to handle the text→audio length mismatch,
    transformer layers for sequence modeling, and a PostNet for refinement.

    Pipeline:
        1. Duration Predictor: How long to speak each token
        2. Length Regulator: Expand text embeddings to audio length
        3. Positional Encoding: Add timing information
        4. Transformer layers: Model audio sequence
        5. Linear projection: d_model → n_mels
        6. PostNet: Refine mel spectrogram quality

    Args:
        config: SpeechDecoderConfig with architecture parameters

    Example:
        decoder = SpeechDecoder(SpeechDecoderConfig(d_model=384))
        text_embeds = torch.randn(2, 20, 384)  # From LIVO
        mel, durations = decoder(text_embeds)
        # mel: (2, ~100, 80) — predicted mel spectrogram
    """

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        self.config = config

        # --- Duration Predictor ---
        self.duration_predictor = DurationPredictor(config.d_model)

        # --- Length Regulator ---
        self.length_regulator = LengthRegulator()

        # --- Positional Encoding for Audio ---
        self.position_embedding = nn.Embedding(config.max_audio_length, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # --- Transformer Layers ---
        # Reuse the SAME TransformerBlock from LIVO.
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

        # --- Mel Projection ---
        self.mel_projection = nn.Linear(config.d_model, config.n_mels)

        # --- PostNet (refinement) ---
        self.postnet = PostNet(
            n_mels=config.n_mels,
            num_layers=config.num_postnet_layers,
            channels=config.postnet_channels,
        )

        self.output_norm = nn.LayerNorm(config.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.mel_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mel_projection.bias)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        max_audio_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert text embeddings to mel spectrogram.

        Args:
            text_embeddings: (batch, text_seq_len, d_model)
                Output from LIVO's transformer (hidden states).
            target_durations: (batch, text_seq_len) — ground truth durations
                Used during training. If None, uses predicted durations.
            max_audio_length: Cap on output mel frames.

        Returns:
            Dict with:
                - mel: (batch, audio_len, n_mels) — predicted mel spectrogram
                - mel_postnet: (batch, audio_len, n_mels) — refined mel
                - durations: (batch, text_seq_len) — predicted durations
        """
        if max_audio_length is None:
            max_audio_length = self.config.max_audio_length

        # 1. Predict durations (how long to speak each token)
        predicted_durations = self.duration_predictor(text_embeddings)

        # 2. Expand text embeddings to audio length
        durations = target_durations if target_durations is not None else predicted_durations
        expanded = self.length_regulator(text_embeddings, durations, max_audio_length)

        # 3. Add positional encoding
        batch_size, seq_len, _ = expanded.shape
        positions = torch.arange(seq_len, device=expanded.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = expanded + self.position_embedding(positions)

        # 4. Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, causal=False)

        x = self.output_norm(x)

        # 5. Project to mel spectrogram
        mel = self.mel_projection(x)  # (batch, audio_len, n_mels)

        # 6. PostNet refinement
        mel_postnet = self.postnet(mel)

        return {
            "mel": mel,
            "mel_postnet": mel_postnet,
            "durations": predicted_durations,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
#  Griffin-Lim Vocoder — Mel Spectrogram → Audio Waveform            #
#  Pure math, no neural network, no training needed.                 #
# ------------------------------------------------------------------ #

class GriffinLimVocoder:
    """
    From-scratch Griffin-Lim vocoder.

    Converts mel spectrograms back to audio waveforms using the
    Griffin-Lim phase reconstruction algorithm. This is a classical
    signal processing approach — no neural network, no training.

    Pipeline:
        Mel Spectrogram → Inverse Mel Filter → STFT magnitude
        → Griffin-Lim iterations → Audio Waveform

    Args:
        sample_rate: Audio sample rate in Hz (default: 22050)
        n_fft: FFT window size (default: 1024)
        hop_length: STFT hop length (default: 256)
        n_mels: Number of mel frequency bins (default: 80)
        num_iterations: Griffin-Lim iterations (default: 60)

    Example:
        vocoder = GriffinLimVocoder()
        mel = model_output["mel_postnet"]  # (1, time, 80)
        audio = vocoder.synthesize(mel)     # numpy array
        vocoder.save_wav(audio, "output.wav")
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        num_iterations: int = 60,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_iterations = num_iterations

        # Build mel filter bank (maps mel bins → linear frequency bins)
        self.mel_basis = self._build_mel_filter(
            sample_rate, n_fft, n_mels, mel_fmin, mel_fmax
        )
        # Pseudo-inverse for mel → linear conversion
        self.mel_basis_inv = np.linalg.pinv(self.mel_basis)

    def _build_mel_filter(
        self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> np.ndarray:
        """Build a mel-scale filter bank from scratch."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Create mel frequency points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])

        # Convert Hz to FFT bin indices
        n_freqs = n_fft // 2 + 1
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Build triangular filters
        filters = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)

        return filters

    def _griffin_lim(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Griffin-Lim phase reconstruction algorithm.

        Given only the magnitude of an STFT, iteratively estimates
        the phase to reconstruct a plausible audio waveform.
        """
        # Start with random phase
        phase = np.exp(2j * np.pi * np.random.random(magnitude.shape))
        complex_spec = magnitude * phase

        for _ in range(self.num_iterations):
            # Inverse STFT → waveform
            audio = self._istft(complex_spec)
            # Forward STFT → get new phase estimate
            complex_spec = self._stft(audio)
            # Keep original magnitude, update phase
            phase = np.exp(1j * np.angle(complex_spec))
            complex_spec = magnitude * phase

        return self._istft(complex_spec)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform (from scratch)."""
        window = np.hanning(self.n_fft)
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length

        frames = np.zeros((self.n_fft, n_frames))
        for i in range(n_frames):
            start = i * self.hop_length
            frames[:, i] = audio[start:start + self.n_fft] * window

        return np.fft.rfft(frames, axis=0)

    def _istft(self, complex_spec: np.ndarray) -> np.ndarray:
        """Inverse Short-Time Fourier Transform (from scratch)."""
        window = np.hanning(self.n_fft)
        n_frames = complex_spec.shape[1]
        audio_length = self.n_fft + (n_frames - 1) * self.hop_length

        audio = np.zeros(audio_length)
        window_sum = np.zeros(audio_length)

        frames = np.fft.irfft(complex_spec, n=self.n_fft, axis=0)

        for i in range(n_frames):
            start = i * self.hop_length
            audio[start:start + self.n_fft] += frames[:, i] * window
            window_sum[start:start + self.n_fft] += window ** 2

        # Normalize by window overlap
        nonzero = window_sum > 1e-8
        audio[nonzero] /= window_sum[nonzero]

        return audio

    def synthesize(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel_spectrogram: (1, time, n_mels) or (time, n_mels)
                Model output mel spectrogram.

        Returns:
            Audio waveform as numpy array (float32, normalized to [-1, 1]).
        """
        # Handle tensor input
        if isinstance(mel_spectrogram, torch.Tensor):
            mel = mel_spectrogram.detach().cpu().numpy()
        else:
            mel = mel_spectrogram

        # Remove batch dimension
        if mel.ndim == 3:
            mel = mel[0]

        # mel shape: (time, n_mels) → transpose to (n_mels, time)
        mel = mel.T

        # Convert mel spectrogram to linear spectrogram
        # Undo log scaling (mel spectrograms are typically in log scale)
        mel_linear = np.exp(mel)

        # Mel → linear frequency using pseudo-inverse
        linear_spec = np.maximum(self.mel_basis_inv @ mel_linear, 1e-10)

        # Griffin-Lim: reconstruct phase and create waveform
        audio = self._griffin_lim(linear_spec)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32)

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        """
        Save audio waveform to a .wav file (from scratch, no scipy needed).

        Args:
            audio: Audio waveform (float32 numpy array)
            path: Output file path (e.g., "output.wav")
        """
        import struct

        # Convert float [-1, 1] to int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        num_channels = 1
        sample_width = 2  # 16-bit
        num_frames = len(audio_int16)
        byte_rate = self.sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width

        # WAV file header (44 bytes) — written from scratch
        header = struct.pack(
            '<4sI4s'       # RIFF header
            '4sIHHIIHH'    # fmt chunk
            '4sI',         # data chunk header
            b'RIFF',
            36 + len(audio_bytes),  # File size - 8
            b'WAVE',
            b'fmt ',
            16,            # fmt chunk size
            1,             # PCM format
            num_channels,
            self.sample_rate,
            byte_rate,
            block_align,
            sample_width * 8,  # Bits per sample
            b'data',
            len(audio_bytes),
        )

        with open(path, 'wb') as f:
            f.write(header)
            f.write(audio_bytes)
