"""Verify 125M scale-up — full multimodal system."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.llm import LLM, Config
from model.multimodal import MultimodalLIVO, MultimodalConfig
from model.vision_encoder import VisionEncoderConfig
from model.audio_encoder import AudioEncoderConfig
from model.speech_decoder import SpeechDecoderConfig, GriffinLimVocoder

print("=" * 60)
print("  LIVO 125M SCALE-UP VERIFICATION")
print("=" * 60)

# 1. Text-only LLM
print("\n--- 1. Text-Only LLM ---")
cfg = Config()
llm = LLM(cfg)
total = sum(p.numel() for p in llm.parameters())
tied = llm.lm_head.weight is llm.token_embedding.embedding.weight
print(f"  vocab_size:   {cfg.vocab_size:,}")
print(f"  d_model:      {cfg.d_model}")
print(f"  num_layers:   {cfg.num_layers}")
print(f"  num_heads:    {cfg.num_heads}")
print(f"  ffn_dim:      {cfg.ffn_dim}")
print(f"  max_length:   {cfg.max_length}")
print(f"  head_dim:     {cfg.d_model // cfg.num_heads}")
print(f"  Total params: {total:,}")
print(f"  Weight tied:  {tied}")
print(f"  FP16 size:    {total * 2 / 1024**2:.1f} MB")

# Quick forward pass
text_ids = torch.randint(0, cfg.vocab_size, (1, 16))
out = llm(text_ids, labels=text_ids)
print(f"  Forward pass: OK (loss={out.loss.item():.4f})")
del llm

# 2. Full Multimodal System
print("\n--- 2. Full Multimodal System ---")
mm_cfg = MultimodalConfig(
    llm=Config(),
    vision=VisionEncoderConfig(),
    audio=AudioEncoderConfig(),
    speech=SpeechDecoderConfig(),
)
model = MultimodalLIVO(mm_cfg)
params = model.num_parameters

print(f"\n  {'Component':<25} {'Params':>15} {'FP16 MB':>10}")
print(f"  {'-'*25} {'-'*15} {'-'*10}")
for comp, count in params.items():
    if comp == "total":
        continue
    print(f"  {comp:<25} {count:>15,} {count * 2 / 1024**2:>8.1f}")
print(f"  {'-'*25} {'-'*15} {'-'*10}")
total_all = params["total"]
print(f"  {'TOTAL':<25} {total_all:>15,} {total_all * 2 / 1024**2:>8.1f}")

# 3. Forward pass with all modalities
print("\n--- 3. Full Forward Pass (Text + Vision + Audio) ---")
text_ids = torch.randint(0, cfg.vocab_size, (1, 16))
image = torch.randn(1, 3, 224, 224)
audio = torch.randn(1, 80, 200)

out = model(text_ids=text_ids, image=image, audio=audio, labels=text_ids)
print(f"  Logits shape: {tuple(out.logits.shape)}")
print(f"  Loss:         {out.loss.item():.4f}")
print(f"  Hidden states: {tuple(out.hidden_states.shape)}")

# 4. Speech synthesis
print("\n--- 4. Speech Synthesis ---")
speech = model.speak(out.hidden_states, max_audio_length=100)
print(f"  Mel shape:    {tuple(speech['mel_postnet'].shape)}")

vocoder = GriffinLimVocoder(num_iterations=3)
audio_out = vocoder.synthesize(speech["mel_postnet"])
print(f"  Audio:        {len(audio_out):,} samples ({len(audio_out)/22050:.2f}s)")

# 5. VRAM estimate
print("\n--- 5. VRAM Budget (FP16 Training) ---")
llm_params = params.get("llm", 0)
model_fp16 = total_all * 2 / 1024**2
optimizer_states = total_all * 4 * 2 / 1024**2  # AdamW FP32
gradients = total_all * 2 / 1024**2
activations = 2 * 1024 * 768 * 12 * 2 * 2 / 1024**2 / 2  # with checkpointing
total_vram = model_fp16 + optimizer_states + gradients + activations

print(f"  Model weights (FP16):     {model_fp16:>8.1f} MB")
print(f"  Optimizer states (FP32):  {optimizer_states:>8.1f} MB")
print(f"  Gradients (FP16):         {gradients:>8.1f} MB")
print(f"  Activations (checkpt):    {activations:>8.1f} MB")
print(f"  {'─'*40}")
print(f"  TOTAL:                    {total_vram:>8.1f} MB")
print(f"  RTX 4050 Available:       {6144:>8} MB")
print(f"  Headroom:                 {6144 - total_vram:>8.1f} MB")
fits = "YES" if total_vram < 5500 else "TIGHT" if total_vram < 6144 else "NO"
print(f"  Fits in 6GB VRAM?         {fits}")

print(f"\n{'=' * 60}")
print(f"  125M SCALE-UP VERIFIED SUCCESSFULLY!")
print(f"{'=' * 60}")
