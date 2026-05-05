"""
Quick test to verify LIVO's multimodal architecture works.

Tests:
  1. Audio encoder standalone
  2. Vision encoder standalone
  3. Multimodal LIVO (all modalities combined)
  4. Parameter counts
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.audio_encoder import AudioEncoder, AudioEncoderConfig
from model.vision_encoder import VisionEncoder, VisionEncoderConfig
from model.multimodal import MultimodalLIVO, MultimodalConfig
from model.llm import Config


def test_audio_encoder():
    print("=" * 50)
    print("TEST 1: Audio Encoder")
    print("=" * 50)

    config = AudioEncoderConfig(d_model=384, num_layers=2)
    encoder = AudioEncoder(config)

    # Simulate a mel spectrogram: (batch=2, n_mels=80, time_frames=300)
    mel = torch.randn(2, 80, 300)
    output = encoder(mel)

    print(f"  Input:      mel spectrogram {tuple(mel.shape)}")
    print(f"  Output:     audio embeddings {tuple(output.shape)}")
    print(f"  Parameters: {encoder.num_parameters:,}")
    print(f"  ✅ Audio Encoder works!\n")


def test_vision_encoder():
    print("=" * 50)
    print("TEST 2: Vision Encoder")
    print("=" * 50)

    config = VisionEncoderConfig(d_model=384, num_layers=3)
    encoder = VisionEncoder(config)

    # Simulate an image: (batch=2, RGB=3, H=224, W=224)
    image = torch.randn(2, 3, 224, 224)
    output = encoder(image)

    print(f"  Input:      image {tuple(image.shape)}")
    print(f"  Output:     vision embeddings {tuple(output.shape)}")
    print(f"  Patches:    {config.num_patches} + 1 CLS = {config.num_patches + 1}")
    print(f"  Parameters: {encoder.num_parameters:,}")

    # Test CLS extraction
    cls_emb = encoder.get_cls_embedding(image)
    print(f"  CLS embed:  {tuple(cls_emb.shape)}")
    print(f"  ✅ Vision Encoder works!\n")


def test_multimodal_text_only():
    print("=" * 50)
    print("TEST 3: Multimodal LIVO (text only)")
    print("=" * 50)

    config = MultimodalConfig(llm=Config(d_model=384, num_layers=2))
    model = MultimodalLIVO(config)

    text_ids = torch.randint(0, 16384, (2, 32))
    labels = text_ids.clone()
    output = model(text_ids=text_ids, labels=labels)

    print(f"  Input:      text_ids {tuple(text_ids.shape)}")
    print(f"  Logits:     {tuple(output.logits.shape)}")
    print(f"  Loss:       {output.loss.item():.4f}")
    print(f"  ✅ Text-only mode works!\n")


def test_multimodal_vision_text():
    print("=" * 50)
    print("TEST 4: Multimodal LIVO (vision + text)")
    print("=" * 50)

    config = MultimodalConfig(
        llm=Config(d_model=384, num_layers=2),
        vision=VisionEncoderConfig(d_model=384, num_layers=2),
    )
    model = MultimodalLIVO(config)

    image = torch.randn(2, 3, 224, 224)
    text_ids = torch.randint(0, 16384, (2, 32))
    labels = text_ids.clone()

    output = model(text_ids=text_ids, image=image, labels=labels)

    print(f"  Image:      {tuple(image.shape)}")
    print(f"  Text:       {tuple(text_ids.shape)}")
    print(f"  Logits:     {tuple(output.logits.shape)}")
    print(f"  Loss:       {output.loss.item():.4f}")

    params = model.num_parameters
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    print(f"  ✅ Vision + Text mode works!\n")


def test_multimodal_all():
    print("=" * 50)
    print("TEST 5: Multimodal LIVO (vision + audio + text)")
    print("=" * 50)

    config = MultimodalConfig(
        llm=Config(d_model=384, num_layers=2),
        vision=VisionEncoderConfig(d_model=384, num_layers=2),
        audio=AudioEncoderConfig(d_model=384, num_layers=2),
    )
    model = MultimodalLIVO(config)

    image = torch.randn(1, 3, 224, 224)
    mel = torch.randn(1, 80, 300)
    text_ids = torch.randint(0, 16384, (1, 32))
    labels = text_ids.clone()

    output = model(text_ids=text_ids, image=image, audio=mel, labels=labels)

    total_seq = output.logits.shape[1]
    print(f"  Image:      {tuple(image.shape)}")
    print(f"  Audio:      {tuple(mel.shape)}")
    print(f"  Text:       {tuple(text_ids.shape)}")
    print(f"  Combined:   {total_seq} total tokens")
    print(f"  Logits:     {tuple(output.logits.shape)}")
    print(f"  Loss:       {output.loss.item():.4f}")

    params = model.num_parameters
    print(f"\n  Parameter Breakdown:")
    for k, v in params.items():
        print(f"    {k}: {v:,}")
    print(f"  ✅ Full multimodal mode works!\n")


if __name__ == "__main__":
    print("\n🧠 LIVO Multimodal Architecture Tests\n")

    test_audio_encoder()
    test_vision_encoder()
    test_multimodal_text_only()
    test_multimodal_vision_text()
    test_multimodal_all()

    print("=" * 50)
    print("🎉 ALL TESTS PASSED — Multimodal LIVO is ready!")
    print("=" * 50)
