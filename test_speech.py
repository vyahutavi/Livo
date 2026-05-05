"""
Test LIVO Speech Decoder + Griffin-Lim Vocoder.

Tests:
  1. Speech decoder (text embeddings → mel spectrogram)
  2. Griffin-Lim vocoder (mel spectrogram → audio waveform)
  3. Full pipeline (text embeddings → .wav file)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.speech_decoder import SpeechDecoder, SpeechDecoderConfig, GriffinLimVocoder


def test_speech_decoder():
    print("=" * 50)
    print("TEST 1: Speech Decoder")
    print("=" * 50)

    config = SpeechDecoderConfig(d_model=384, num_layers=2)
    decoder = SpeechDecoder(config)

    # Simulate text embeddings from LIVO (20 tokens, 384-dim)
    text_embeds = torch.randn(2, 20, 384)
    output = decoder(text_embeds)

    print(f"  Input:          text embeddings {tuple(text_embeds.shape)}")
    print(f"  Mel output:     {tuple(output['mel'].shape)}")
    print(f"  Mel (PostNet):  {tuple(output['mel_postnet'].shape)}")
    print(f"  Durations:      {tuple(output['durations'].shape)}")
    print(f"  Duration range: {output['durations'].min().item():.2f} - {output['durations'].max().item():.2f}")
    print(f"  Parameters:     {decoder.num_parameters:,}")
    print(f"  ✅ Speech Decoder works!\n")


def test_griffin_lim_vocoder():
    print("=" * 50)
    print("TEST 2: Griffin-Lim Vocoder")
    print("=" * 50)

    vocoder = GriffinLimVocoder(num_iterations=10)  # fewer iterations for speed

    # Simulate a mel spectrogram (100 frames, 80 mels)
    mel = torch.randn(1, 100, 80) * 0.5

    audio = vocoder.synthesize(mel)

    print(f"  Input:          mel spectrogram (1, 100, 80)")
    print(f"  Output:         audio waveform ({len(audio):,} samples)")
    print(f"  Duration:       {len(audio) / vocoder.sample_rate:.2f} seconds")
    print(f"  Sample rate:    {vocoder.sample_rate} Hz")
    print(f"  Audio range:    [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  ✅ Griffin-Lim Vocoder works!\n")


def test_full_pipeline():
    print("=" * 50)
    print("TEST 3: Full Pipeline (embeddings → .wav)")
    print("=" * 50)

    # 1. Speech Decoder
    config = SpeechDecoderConfig(d_model=384, num_layers=2)
    decoder = SpeechDecoder(config)

    text_embeds = torch.randn(1, 15, 384)
    output = decoder(text_embeds, max_audio_length=200)

    # 2. Vocoder
    vocoder = GriffinLimVocoder(num_iterations=10)
    mel = output["mel_postnet"]
    audio = vocoder.synthesize(mel)

    # 3. Save to WAV
    output_path = str(PROJECT_ROOT / "checkpoints" / "test_speech.wav")
    vocoder.save_wav(audio, output_path)

    print(f"  Text embeddings: (1, 15, 384)")
    print(f"  Mel spectrogram: {tuple(mel.shape)}")
    print(f"  Audio waveform:  {len(audio):,} samples")
    print(f"  Duration:        {len(audio) / vocoder.sample_rate:.2f} seconds")
    print(f"  Saved to:        {output_path}")
    print(f"  ✅ Full pipeline works!\n")


if __name__ == "__main__":
    print("\n🔊 LIVO Speech Output Tests\n")

    test_speech_decoder()
    test_griffin_lim_vocoder()
    test_full_pipeline()

    print("=" * 50)
    print("🎉 ALL SPEECH TESTS PASSED — LIVO can speak!")
    print("=" * 50)
