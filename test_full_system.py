"""Test full multimodal + speech integration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.multimodal import MultimodalLIVO, MultimodalConfig
from model.llm import Config
from model.speech_decoder import SpeechDecoderConfig, GriffinLimVocoder
from model.vision_encoder import VisionEncoderConfig
from model.audio_encoder import AudioEncoderConfig

print("=" * 55)
print("TEST: Full Multimodal + Speech (all modalities)")
print("=" * 55)

config = MultimodalConfig(
    llm=Config(d_model=384, num_layers=2),
    vision=VisionEncoderConfig(d_model=384, num_layers=2),
    audio=AudioEncoderConfig(d_model=384, num_layers=2),
    speech=SpeechDecoderConfig(d_model=384, num_layers=2),
)
model = MultimodalLIVO(config)

image = torch.randn(1, 3, 224, 224)
mel_input = torch.randn(1, 80, 300)
text_ids = torch.randint(0, 16384, (1, 20))

# Forward: image + audio + text -> logits + loss + hidden_states
out = model(text_ids=text_ids, image=image, audio=mel_input, labels=text_ids)

# Speech: hidden_states -> mel spectrogram -> audio waveform
speech = model.speak(out.hidden_states, max_audio_length=200)

# Vocoder: mel -> .wav
vocoder = GriffinLimVocoder(num_iterations=5)
audio = vocoder.synthesize(speech["mel_postnet"])
vocoder.save_wav(audio, str(PROJECT_ROOT / "checkpoints" / "test_full.wav"))

print(f"  Logits:      {tuple(out.logits.shape)}")
print(f"  Loss:        {out.loss.item():.4f}")
print(f"  Speech mel:  {tuple(speech['mel_postnet'].shape)}")
print(f"  Audio:       {len(audio):,} samples ({len(audio)/22050:.2f}s)")
print(f"  Saved:       checkpoints/test_full.wav")
print()

params = model.num_parameters
print("  Parameter Breakdown:")
for k, v in params.items():
    print(f"    {k}: {v:,}")

print()
print("  " + "=" * 53)
print("  FULL MULTIMODAL + SPEECH TEST PASSED!")
print("  " + "=" * 53)
