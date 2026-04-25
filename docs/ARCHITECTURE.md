# LIVO Architecture — Technical Deep Dive

## 1. High-Level Architecture

LIVO is a **decoder-only causal language model** — the same family as GPT-2, LLaMA, and Gemma. It processes text autoregressively: given a sequence of tokens, it predicts the next token at every position.

```
"Once upon a" → Model → predicts "time"
"Once upon a time" → Model → predicts "there"
...
```

### Why Decoder-Only (not Encoder-Decoder)?

| Aspect | Decoder-Only (LIVO) | Encoder-Decoder (T5/BART) |
|---|---|---|
| **Attention** | Causal (each token sees only past) | Encoder: bidirectional, Decoder: causal + cross-attention |
| **Use case** | Text generation, completion | Translation, summarization, Q&A |
| **Parameters** | Single stack (efficient) | Two stacks (more params) |
| **Modern trend** | GPT-4, LLaMA, Gemma, Mistral | T5, mBART, FLAN |

For story generation on TinyStories, decoder-only is the natural and efficient choice.

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
- **Deterministic defaults**: seed corpus provides meaningful initial merges
- **Spike encoding**: experimental hash-based binary representation for neural spike models

### 2.2 Embeddings (model/embeddings.py)

Two embedding layers that convert token IDs into dense vectors:

```python
# Token Embedding: token_id → vector
TokenEmbedding(16384, 384)  # ~6.3M parameters

# Position Embedding: position → vector  
LearnedPositionEmbedding(512, 384)  # ~197K parameters

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
- Saves ~6.3M parameters (would be ~23.4M without tying)
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

### 3.1 Data Flow

```
TinyStories (HuggingFace)
    │
    ▼
TinyStoriesDataset
    │  tokenize with livorator
    │  pad to 512 tokens
    │  create attention_mask
    ▼
CausalLMCollator
    │  stack into batches
    ▼
DataLoader (batch_size=2, shuffle=True)
    │
    ▼
Trainer
    │  forward → loss → backward → optimizer step
    ▼
Checkpoints (saved every 500 steps)
```

### 3.2 Loss Function

**Shifted Cross-Entropy** — the standard causal LM training objective:

```
Input:   [<bos>  Once  upon  a    time  <eos>  <pad>]
Labels:  [Once   upon  a     time <eos>  <pad>  <pad>]
```

The model learns to predict the next token at every position. `<pad>` tokens are ignored in the loss (`ignore_index=0`).

### 3.3 Optimizer & Schedule

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

### 3.4 Gradient Accumulation

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

## 4. Parameter Count Breakdown

| Component | Parameters | % of Total |
|---|---|---|
| Token Embedding (shared with LM Head) | 6,291,456 | 36.7% |
| Position Embedding | 196,608 | 1.1% |
| Transformer Blocks (×6) | 10,646,784 | 62.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;├ Attention (per layer) | 591,360 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;├ FFN (per layer) | 1,181,568 | — |
| &nbsp;&nbsp;&nbsp;&nbsp;└ LayerNorms (per layer) | 1,536 | — |
| Final LayerNorm | 768 | ~0% |
| **Total** | **17,135,616** | **100%** |

> Note: LM Head parameters are not counted separately because they are weight-tied with Token Embedding.

---

## 5. Inference / Generation

Text generation is autoregressive — one token at a time:

```
Step 1: input = [<bos>]           → model predicts "Once"
Step 2: input = [<bos>, Once]     → model predicts "upon"
Step 3: input = [<bos>, Once, upon] → model predicts "a"
...until <eos> or max_new_tokens reached
```

**Sampling strategies** (implemented in `livorator`):

| Strategy | Effect |
|---|---|
| **Temperature** (0.8) | Lower = more deterministic, Higher = more creative |
| **Top-k** (50) | Only consider top 50 most likely tokens |
| **Top-p / Nucleus** (0.9) | Only consider tokens within 90% cumulative probability |

---

## 6. Design Decisions & Trade-offs

| Decision | Choice | Alternative | Rationale |
|---|---|---|---|
| Architecture | Decoder-only | Encoder-decoder | Simpler, efficient for generation |
| Normalization | Pre-norm | Post-norm | More stable training, used by GPT-2+ |
| Position encoding | Learned | RoPE / Sinusoidal | Simpler, works well for 512 context |
| Activation | GELU | SiLU / ReLU | Standard for transformers since GPT-2 |
| Attention | PyTorch MHA | Custom / Flash Attention | Simpler, correct, good enough for 17M |
| Tokenizer | Custom BPE | SentencePiece / tiktoken | Educational, full control |
| Weight tying | Enabled | Disabled | Saves 6.3M params, proven effective |
| Gradient checkpointing | Enabled | Disabled | Saves ~50% memory, slight speed cost |
