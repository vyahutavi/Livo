"""
LIVO Parameter Calculator — Detailed breakdown of every single parameter.
Calculates exact numbers for each layer, each component, and the full system.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.llm import LLM, Config
from model.audio_encoder import AudioEncoder, AudioEncoderConfig
from model.vision_encoder import VisionEncoder, VisionEncoderConfig
from model.speech_decoder import SpeechDecoder, SpeechDecoderConfig
from model.multimodal import MultimodalLIVO, MultimodalConfig


def count_params(module):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_num(n):
    """Format number with commas."""
    return f"{n:,}"


def format_mb(n):
    """Convert param count to MB (assuming float32 = 4 bytes)."""
    return f"{n * 4 / (1024**2):.2f} MB"


def format_mb_fp16(n):
    """Convert param count to MB (assuming float16 = 2 bytes)."""
    return f"{n * 2 / (1024**2):.2f} MB"


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_section(title):
    print(f"\n  --- {title} ---")


def detailed_breakdown(model, name="Model"):
    """Print every named parameter with shape and count."""
    print(f"\n  {'Layer':<50} {'Shape':<22} {'Params':>12}")
    print(f"  {'-'*50} {'-'*22} {'-'*12}")
    
    total = 0
    for pname, param in model.named_parameters():
        count = param.numel()
        total += count
        shape_str = str(tuple(param.shape))
        print(f"  {pname:<50} {shape_str:<22} {format_num(count):>12}")
    
    print(f"  {'-'*50} {'-'*22} {'-'*12}")
    print(f"  {'TOTAL':<50} {'':<22} {format_num(total):>12}")
    return total


# ================================================================
# 1. CORE LLM (Text Model)
# ================================================================
print_header("1. CORE LLM — Text Model (Default Config)")

config = Config()
llm = LLM(config)

print(f"\n  Config:")
print(f"    vocab_size:  {config.vocab_size:,}")
print(f"    d_model:     {config.d_model}")
print(f"    num_layers:  {config.num_layers}")
print(f"    num_heads:   {config.num_heads}")
print(f"    ffn_dim:     {config.ffn_dim}")
print(f"    max_length:  {config.max_length}")
print(f"    head_dim:    {config.d_model // config.num_heads}")
print(f"    dropout:     {config.dropout}")

print_section("Manual Calculation")

# Token Embedding
tok_emb = config.vocab_size * config.d_model
print(f"  Token Embedding:      {config.vocab_size} × {config.d_model} = {format_num(tok_emb)}")

# Position Embedding
pos_emb = config.max_length * config.d_model
print(f"  Position Embedding:   {config.max_length} × {config.d_model} = {format_num(pos_emb)}")

# Per Transformer Block
d = config.d_model
h = config.num_heads
f = config.ffn_dim

# Multi-Head Attention: Q, K, V projections + output projection
# nn.MultiheadAttention stores: in_proj_weight (3*d, d) + in_proj_bias (3*d) + out_proj.weight (d, d) + out_proj.bias (d)
qkv_weight = 3 * d * d
qkv_bias = 3 * d
out_weight = d * d
out_bias = d
attn_total = qkv_weight + qkv_bias + out_weight + out_bias
print(f"\n  Per Transformer Block:")
print(f"    Attention QKV weight: 3 × {d} × {d} = {format_num(qkv_weight)}")
print(f"    Attention QKV bias:   3 × {d} = {format_num(qkv_bias)}")
print(f"    Attention out weight: {d} × {d} = {format_num(out_weight)}")
print(f"    Attention out bias:   {d} = {format_num(out_bias)}")
print(f"    Attention subtotal:   {format_num(attn_total)}")

# FFN: Linear(d, ffn_dim) + bias + Linear(ffn_dim, d) + bias
ffn_w1 = d * f
ffn_b1 = f
ffn_w2 = f * d
ffn_b2 = d
ffn_total = ffn_w1 + ffn_b1 + ffn_w2 + ffn_b2
print(f"    FFN up weight:        {d} × {f} = {format_num(ffn_w1)}")
print(f"    FFN up bias:          {f} = {format_num(ffn_b1)}")
print(f"    FFN down weight:      {f} × {d} = {format_num(ffn_w2)}")
print(f"    FFN down bias:        {d} = {format_num(ffn_b2)}")
print(f"    FFN subtotal:         {format_num(ffn_total)}")

# LayerNorms: 2 per block, each has weight (d) + bias (d)
ln_per_block = 2 * (d + d)
print(f"    LayerNorms (×2):      2 × ({d} + {d}) = {format_num(ln_per_block)}")

block_total = attn_total + ffn_total + ln_per_block
print(f"    Block total:          {format_num(block_total)}")

all_blocks = block_total * config.num_layers
print(f"    All {config.num_layers} blocks:          {format_num(all_blocks)}")

# Final LayerNorm
final_ln = d + d
print(f"\n  Final LayerNorm:        {d} + {d} = {format_num(final_ln)}")

# LM Head (weight-tied with token embedding, so NOT counted separately)
lm_head_note = "SHARED (weight-tied with Token Embedding)"
print(f"  LM Head:                {lm_head_note}")

# Total (unique parameters)
llm_total_calc = tok_emb + pos_emb + all_blocks + final_ln
print(f"\n  Calculated Total:       {format_num(llm_total_calc)}")

print_section("Actual PyTorch Parameters (Verified)")
llm_total_actual = detailed_breakdown(llm, "LLM")

# Verify
total_params, trainable_params = count_params(llm)
print(f"\n  Total params (PyTorch):     {format_num(total_params)}")
print(f"  Trainable params:           {format_num(trainable_params)}")
print(f"  Memory (FP32):              {format_mb(total_params)}")
print(f"  Memory (FP16):              {format_mb_fp16(total_params)}")

# Check weight tying
tied = llm.lm_head.weight is llm.token_embedding.embedding.weight
print(f"  Weight tying active:        {'YES' if tied else 'NO'}")
if tied:
    without_tying = total_params + tok_emb
    print(f"  Params WITHOUT tying:       {format_num(without_tying)}")
    print(f"  Params saved by tying:      {format_num(tok_emb)}")


# ================================================================
# 2. AUDIO ENCODER
# ================================================================
print_header("2. AUDIO ENCODER")

audio_cfg = AudioEncoderConfig(d_model=384, num_layers=2)
audio_enc = AudioEncoder(audio_cfg)

print(f"\n  Config:")
print(f"    n_mels:      {audio_cfg.n_mels}")
print(f"    d_model:     {audio_cfg.d_model}")
print(f"    num_layers:  {audio_cfg.num_layers}")
print(f"    conv_channels: {audio_cfg.conv_channels}")

print_section("Actual PyTorch Parameters")
audio_total = detailed_breakdown(audio_enc, "AudioEncoder")

total_p, _ = count_params(audio_enc)
print(f"\n  Total params:       {format_num(total_p)}")
print(f"  Memory (FP32):      {format_mb(total_p)}")
print(f"  Memory (FP16):      {format_mb_fp16(total_p)}")


# ================================================================
# 3. VISION ENCODER
# ================================================================
print_header("3. VISION ENCODER")

vision_cfg = VisionEncoderConfig(d_model=384, num_layers=3)
vision_enc = VisionEncoder(vision_cfg)

print(f"\n  Config:")
print(f"    img_size:    {vision_cfg.img_size}")
print(f"    patch_size:  {vision_cfg.patch_size}")
print(f"    num_patches: {vision_cfg.num_patches}")
print(f"    patch_dim:   {vision_cfg.patch_dim}")
print(f"    d_model:     {vision_cfg.d_model}")
print(f"    num_layers:  {vision_cfg.num_layers}")
print(f"    CLS token:   {vision_cfg.use_cls_token}")

print_section("Actual PyTorch Parameters")
vision_total = detailed_breakdown(vision_enc, "VisionEncoder")

total_p, _ = count_params(vision_enc)
print(f"\n  Total params:       {format_num(total_p)}")
print(f"  Memory (FP32):      {format_mb(total_p)}")
print(f"  Memory (FP16):      {format_mb_fp16(total_p)}")


# ================================================================
# 4. SPEECH DECODER
# ================================================================
print_header("4. SPEECH DECODER")

speech_cfg = SpeechDecoderConfig(d_model=384, num_layers=2)
speech_dec = SpeechDecoder(speech_cfg)

print(f"\n  Config:")
print(f"    d_model:            {speech_cfg.d_model}")
print(f"    n_mels:             {speech_cfg.n_mels}")
print(f"    num_layers:         {speech_cfg.num_layers}")
print(f"    max_audio_length:   {speech_cfg.max_audio_length}")
print(f"    postnet_layers:     {speech_cfg.num_postnet_layers}")
print(f"    postnet_channels:   {speech_cfg.postnet_channels}")

print_section("Actual PyTorch Parameters")
speech_total = detailed_breakdown(speech_dec, "SpeechDecoder")

total_p, _ = count_params(speech_dec)
print(f"\n  Total params:       {format_num(total_p)}")
print(f"  Memory (FP32):      {format_mb(total_p)}")
print(f"  Memory (FP16):      {format_mb_fp16(total_p)}")


# ================================================================
# 5. FULL MULTIMODAL SYSTEM
# ================================================================
print_header("5. FULL MULTIMODAL LIVO (ALL COMPONENTS)")

mm_config = MultimodalConfig(
    llm=Config(),
    vision=VisionEncoderConfig(d_model=384, num_layers=3),
    audio=AudioEncoderConfig(d_model=384, num_layers=2),
    speech=SpeechDecoderConfig(d_model=384, num_layers=2),
)
mm_model = MultimodalLIVO(mm_config)

params = mm_model.num_parameters

print(f"\n  Component Breakdown:")
print(f"  {'Component':<25} {'Params':>15} {'Memory (FP32)':>15} {'Memory (FP16)':>15} {'% of Total':>12}")
print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

total_all = params["total"]
for component, count in params.items():
    if component == "total":
        continue
    pct = count / total_all * 100
    print(f"  {component:<25} {format_num(count):>15} {format_mb(count):>15} {format_mb_fp16(count):>15} {pct:>10.1f}%")

print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
print(f"  {'TOTAL':<25} {format_num(total_all):>15} {format_mb(total_all):>15} {format_mb_fp16(total_all):>15} {'100.0%':>12}")


# ================================================================
# 6. COMPARISON TABLE
# ================================================================
print_header("6. SIZE COMPARISON WITH KNOWN MODELS")

comparisons = [
    ("LIVO Text-Only",       total_params),
    ("LIVO Full Multimodal", total_all),
    ("GPT-2 Small",          124_000_000),
    ("GPT-2 Medium",         355_000_000),
    ("GPT-2 Large",          774_000_000),
    ("LLaMA-7B",             6_738_000_000),
    ("LLaMA-13B",            13_015_000_000),
    ("GPT-4 (estimated)",    1_800_000_000_000),
]

print(f"\n  {'Model':<25} {'Parameters':>18} {'FP32 Size':>15} {'FP16 Size':>15}")
print(f"  {'-'*25} {'-'*18} {'-'*15} {'-'*15}")
for name, p in comparisons:
    marker = " ◄── YOU" if "LIVO" in name else ""
    print(f"  {name:<25} {format_num(p):>18} {format_mb(p):>15} {format_mb_fp16(p):>15}{marker}")


# ================================================================
# 7. TRAINING COMPUTE ESTIMATES
# ================================================================
print_header("7. TRAINING COMPUTE ESTIMATES")

# TinyStories dataset size
dataset_size = 2_119_719  # approximate total stories
seq_len = 512
batch_size = 2
grad_accum = 4
effective_batch = batch_size * grad_accum
tokens_per_step = effective_batch * seq_len
total_tokens = dataset_size * seq_len  # approximate

# One epoch
steps_per_epoch = dataset_size // effective_batch
tokens_per_epoch = steps_per_epoch * tokens_per_step

print(f"\n  Dataset: TinyStories")
print(f"    Total stories:           {format_num(dataset_size)}")
print(f"    Sequence length:         {seq_len} tokens")
print(f"    Approx total tokens:     {format_num(total_tokens)}")
print(f"")
print(f"  Training Config:")
print(f"    Batch size:              {batch_size}")
print(f"    Gradient accumulation:   {grad_accum}")
print(f"    Effective batch size:    {effective_batch}")
print(f"    Tokens per step:         {format_num(tokens_per_step)}")
print(f"    Steps per epoch:         {format_num(steps_per_epoch)}")
print(f"    Tokens per epoch:        {format_num(tokens_per_epoch)}")

# FLOPs estimate: ~6 * params * tokens (standard transformer estimate)
flops_per_token = 6 * total_params
flops_per_epoch = flops_per_token * tokens_per_epoch
tflops_per_epoch = flops_per_epoch / 1e12

# RTX 4050 Laptop: ~5.7 TFLOPS FP32, ~11.4 TFLOPS FP16
gpu_tflops_fp16 = 11.4
seconds_per_epoch = flops_per_epoch / (gpu_tflops_fp16 * 1e12)
hours_per_epoch = seconds_per_epoch / 3600

print(f"")
print(f"  Compute (Text-Only LLM, {format_num(total_params)} params):")
print(f"    FLOPs per token:         {format_num(flops_per_token)}")
print(f"    FLOPs per epoch:         {flops_per_epoch:.2e}")
print(f"    TFLOPs per epoch:        {tflops_per_epoch:,.1f}")
print(f"")
print(f"  RTX 4050 Laptop (~11.4 TFLOPS FP16):")
print(f"    Est. time per epoch:     {hours_per_epoch:.1f} hours")
print(f"    Est. time for 3 epochs:  {hours_per_epoch * 3:.1f} hours")


# ================================================================
# 8. VRAM USAGE ESTIMATE
# ================================================================
print_header("8. VRAM USAGE ESTIMATE (RTX 4050, 6GB)")

# Model weights
model_fp16 = total_params * 2 / (1024**2)
# Optimizer states (AdamW: 2 states per param)
optimizer_states = total_params * 4 * 2 / (1024**2)  # FP32 states
# Gradients
gradients = total_params * 2 / (1024**2)  # FP16
# Activations (rough estimate: batch * seq * d_model * num_layers * 2)
activations = batch_size * seq_len * config.d_model * config.num_layers * 2 * 2 / (1024**2)
# With gradient checkpointing (~50% reduction)
activations_ckpt = activations / 2

total_vram = model_fp16 + optimizer_states + gradients + activations_ckpt

print(f"\n  {'Component':<30} {'FP16 (MB)':>12} {'Notes':>30}")
print(f"  {'-'*30} {'-'*12} {'-'*30}")
print(f"  {'Model weights':<30} {model_fp16:>10.1f}   FP16 mixed precision")
print(f"  {'Optimizer states (AdamW)':<30} {optimizer_states:>10.1f}   FP32 (momentum + variance)")
print(f"  {'Gradients':<30} {gradients:>10.1f}   FP16")
print(f"  {'Activations (checkpointed)':<30} {activations_ckpt:>10.1f}   ~50% saved by checkpointing")
print(f"  {'-'*30} {'-'*12}")
print(f"  {'ESTIMATED TOTAL':<30} {total_vram:>10.1f}   MB")
print(f"  {'GPU Available':<30} {'6,144':>12}   MB (RTX 4050)")
print(f"  {'Headroom':<30} {6144 - total_vram:>10.1f}   MB")
print(f"")
fits = "YES — comfortably" if total_vram < 5000 else ("YES — tight" if total_vram < 6000 else "NO — reduce batch size")
print(f"  Fits in VRAM? {fits}")


print(f"\n{'=' * 70}")
print(f"  CALCULATION COMPLETE")
print(f"{'=' * 70}\n")
