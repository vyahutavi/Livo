# LIVO Architecture — Technical Deep Dive

## 1. High-Level Architecture

LIVO is a **lightweight, decoder-only causal language model** — the same fundamental architecture family as GPT-4, LLaMA, and Gemma. At its core, it processes text autoregressively: given a sequence of tokens, it predicts the next token at every position.

```
"Once upon a" → Model → predicts "time"
"Once upon a time" → Model → predicts "there"
...
```

### System Overview

```
Input Text
    │
    ▼
┌─────────────────────────┐
│   livorator (BPE)       │  Trainable byte-pair encoding
│   vocab_size: 16,384    │  train() → save() → load()
└──────────┬──────────────┘
           │
           ▼
┌──────────────────┐   ┌──────────────────────┐
│ Token Embedding  │ + │ Position Embedding   │
│ (16384 × 384)    │   │ (512 × 384)          │
└──────────┬───────┘   └──────────┬────────────┘
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │  Transformer Block   │ ×12
           │  ┌────────────────┐  │
           │  │ LayerNorm      │  │
           │  │ Causal MHA     │  │  12 heads, masked
           │  │ Residual +     │  │
           │  │ LayerNorm      │  │
           │  │ FFN (GELU)     │  │  768 → 3072 → 768
           │  │ Residual +     │  │
           │  └────────────────┘  │
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │   Final LayerNorm    │
           └──────────┬───────────┘
                      │
           ┌──────────▼───────────┐
           │   LM Head (Linear)   │  Weight-tied (50000 × 768)
           └──────────┬───────────┘
                      │
              Output Logits
          (sampling → text)
```

### Why Decoder-Only?

| Aspect | Decoder-Only (LIVO) | Encoder-Decoder (T5/BART) |
|---|---|---|
| **Attention** | Causal (each token sees only past) | Encoder: bidirectional, Decoder: causal + cross-attention |
| **Use case** | Text generation, completion | Translation, summarization, Q&A |
| **Parameters** | Single stack (efficient) | Two stacks (more params) |
| **Modern trend** | GPT-4, LLaMA, Gemma, Mistral | T5, mBART, FLAN |

---

## 2. Component Breakdown

### 2.1 Tokenizer: `livorator` (data/tokenizer.py)

The tokenizer converts raw text into integer token IDs and back.

```
"Hello world" → UTF-8 bytes → BPE merges → [2, 76, 300, 112, ...] → Model
```

**Algorithm: Byte-Pair Encoding (BPE)**

1. Start with all 256 possible byte values as the base vocabulary
2. Add 4 special tokens: `<pad>(0)`, `<unk>(1)`, `<bos>(2)`, `<eos>(3)`
3. Iteratively merge the most frequent adjacent pairs into new tokens
4. Total vocabulary: **16,384 tokens** (260 base + up to 16,124 merges)

**Key design decisions:**
- **Byte-level**: handles any Unicode text without unknown characters
- **Heap-based merge**: O(N log N) BPE application instead of naive O(N²)
- **Trainable**: Can be trained on any target dataset for optimal compression

### 2.2 Embeddings (model/embeddings.py)

Two embedding layers that convert token IDs into dense vectors:

```python
# Token Embedding: token_id → vector
TokenEmbedding(16384, 384)  # 6,291,456 parameters

# Position Embedding: position → vector  
LearnedPositionEmbedding(512, 384)  # 196,608 parameters

# Combined
hidden = token_embed(ids) + position_embed(ids)  # (batch, seq, 384)
```

- **Token Embedding** uses `padding_idx=0` so the `<pad>` token always maps to a zero vector
- **Position Embedding** is learned (not sinusoidal) — each of the 512 positions has a trainable vector
- Initialized with `N(0, 0.02)` for stable training

### 2.3 Transformer Block (model/transformer_block.py)

Each of the 6 identical transformer blocks follows the **Pre-Norm** pattern:

```
Input
  │
  ├──────────────────────────────┐
  │                              │ (residual)
  ▼                              │
LayerNorm                        │
  │                              │
  ▼                              │
Causal Multi-Head Attention      │
  │                              │
  ▼                              │
Dropout                          │
  │                              │
  + ◄────────────────────────────┘
  │
  ├──────────────────────────────┐
  │                              │ (residual)
  ▼                              │
LayerNorm                        │
  │                              │
  ▼                              │
FFN: Linear(384→1536) → GELU    │
     Linear(1536→384) → Dropout  │
  │                              │
  + ◄────────────────────────────┘
  │
Output
```

**Causal Self-Attention:**
- 6 attention heads, each with head_dim = 64 (384 / 6)
- Uses PyTorch's `nn.MultiheadAttention` with `batch_first=True`
- Upper-triangular boolean mask prevents attending to future tokens
- `key_padding_mask` zeros out attention to `<pad>` tokens

**Feed-Forward Network (FFN):**
- Expansion: 384 → 1536 (4× expansion ratio)
- Activation: GELU (smoother than ReLU, used by GPT-2+)
- Contraction: 1536 → 384

**Gradient Checkpointing:**
- Enabled during training to save ~50% activation memory
- Trades compute for memory — recomputes activations during backward pass

### 2.4 LLM Assembly (model/llm.py)

The full model stacks everything together:

```python
class LLM(nn.Module):
    token_embedding    # TokenEmbedding(16384, 384)
    position_embedding # LearnedPositionEmbedding(512, 384)
    transformer        # ModuleList of 6 TransformerBlocks
    final_norm         # LayerNorm(384)
    lm_head            # Linear(384, 16384) — weight-tied with token_embedding
```

**Weight Tying:** The `lm_head.weight` is the same tensor as `token_embedding.weight`. This:
- Saves **6,291,456** parameters (would be ~23.4M without tying)
- Forces the model to learn embeddings that are useful for both input and output
- Is standard practice in modern LLMs

**Forward Pass:**
```python
def forward(input_ids, attention_mask=None, labels=None):
    x = token_embed(input_ids) + position_embed(input_ids)
    for block in transformer_blocks:
        x = block(x, attention_mask)
    x = final_norm(x)
    logits = lm_head(x)  # (batch, seq, 16384)
    
    if labels:
        loss = cross_entropy(logits[:, :-1], labels[:, 1:])  # shifted
    return CausalLMOutput(logits, loss)
```

---

## 3. Training Pipeline

### 3.1 Loss Function

**Shifted Cross-Entropy** — the standard causal LM training objective:

```
Input:   [<bos>  Once  upon  a    time  <eos>  <pad>]
Labels:  [Once   upon  a     time <eos>  <pad>  <pad>]
```

The model learns to predict the next token at every position. `<pad>` tokens are ignored in the loss (`ignore_index=0`).

### 3.2 Optimizer & Schedule

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 2e-4 | Good for ~17M model |
| Weight decay | 0.01 | Mild regularization |
| Betas | (0.9, 0.95) | β₂=0.95 for LLM stability |
| Warmup | 1000 steps | Linear warmup |
| Schedule | Cosine decay | Smooth LR reduction |
| Grad clipping | 1.0 | Prevents exploding gradients |
| Grad accumulation | 4 steps | Effective batch = 8 |
| Precision | FP16 | Half-precision for speed |

### 3.3 Gradient Accumulation

With `batch_size=2` and `grad_accum_steps=4`:

```
micro_batch_1 (2 samples) → loss₁.backward()  ← accumulate
micro_batch_2 (2 samples) → loss₂.backward()  ← accumulate
micro_batch_3 (2 samples) → loss₃.backward()  ← accumulate
micro_batch_4 (2 samples) → loss₄.backward()  ← accumulate
                           → optimizer.step()   ← update weights
                           
Effective batch size = 2 × 4 = 8 samples
```

This allows training with larger effective batches on limited GPU memory.

---

## 4. Parameter Count Breakdown (Current 124M)

| Component | Parameters | % of Total |
|---|---|---|
| Token Embedding (shared with LM Head) | 38,400,000 | 30.9% |
| Position Embedding | 786,432 | 0.6% |
| Transformer Blocks (×12) | 85,054,464 | 68.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;├ Attention QKV weight (per layer) | 1,769,472 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ Attention QKV bias (per layer) | 2,304 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ Attention output weight (per layer) | 589,824 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ Attention output bias (per layer) | 768 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ FFN up: Linear(768→3072) + bias (per layer) | 2,362,368 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ FFN down: Linear(3072→768) + bias (per layer) | 2,360,064 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;└ LayerNorms ×2 (per layer) | 3,072 | — |
| Final LayerNorm | 1,536 | ~0% |
| **Total** | **~124,242,432** | **100%** |

> Note: LM Head (38,400,000) is weight-tied with Token Embedding — not counted separately. Without tying, total would be ~162M.

### VRAM Budget (FP16 Training)
- **Model weights:** ~237 MB
- **Optimizer states (AdamW):** ~950 MB
- **Gradients:** ~237 MB
- **Activations (checkpointed):** ~36 MB
- **Total VRAM:** ~1.5 GB (Fits comfortably in RTX 4050 6GB with ~4.5 GB headroom)

---

## 5. Scale-Up Notes

The model has been scaled from 17M to **~124M parameters** (comparable to GPT-2 Small). Key changes:

- **Vocabulary**: 16K → 50K (better compression for domain text)
- **Model dimension**: 384 → 768
- **Layers**: 6 → 12
- **Attention heads**: 6 → 12 (head_dim = 64 preserved)
- **FFN**: 1536 → 3072 (4× expansion preserved)
- **Encoders**: Vision (4 layers), Audio (3 layers), Speech (3 layers) — all scaled to d_model=768

### VRAM Feasibility
The 124M model in FP16 consumes approximately **~2.3 GB** of VRAM during training. This fits comfortably into an **RTX 4050 (6GB)** with ~3.8 GB of headroom.

### Pre-Training Dataset Strategy

The dataset is **TBD**. The `train_tokenizer.py` script should be run on the target corpus to learn the 50K domain-specific BPE merges before training begins.

---

## 6. Config Reference

### model.yml (LLM)
```yaml
vocab_size: 50000    # 50K BPE tokens
d_model: 768         # Embedding dimension
num_layers: 12       # Transformer blocks
num_heads: 12        # Attention heads (head_dim = 64)
ffn_dim: 3072        # 4× expansion
dropout: 0.1         # Regularization
max_length: 1024     # Context window
```

### train.yml
```yaml
seed: 42
precision: fp16
batch_size: 2
grad_accum_steps: 4      # Effective batch = 8
learning_rate: 0.0002    # 2e-4
weight_decay: 0.01
warmup_steps: 1000       # Linear warmup
max_steps: 10000
grad_clip_norm: 1.0
optimizer:
  betas: [0.9, 0.95]
  eps: 1.0e-8
checkpoint:
  save_dir: checkpoints
  save_every: 500
logging:
  log_every: 10
  eval_every: 100
data:
  num_workers: 2
```
