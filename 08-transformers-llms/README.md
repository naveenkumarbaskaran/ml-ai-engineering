# Transformers & LLMs

## Self-Attention: The Key Innovation ⭐

```
"Attention Is All You Need" (Vaswani et al., 2017)

Input: "The cat sat on the mat"

For each word, compute:
  Q (Query): "What am I looking for?"
  K (Key):   "What do I contain?"
  V (Value): "What do I provide?"

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

### Why √dₖ?
- Without scaling, dot products grow with dimension → softmax saturates → vanishing gradients
- Dividing by √dₖ keeps variance ≈ 1

### Multi-Head Attention
```
Instead of one attention: split into h heads
  head_i = Attention(Q·WᵢQ, K·WᵢK, V·WᵢV)
  MultiHead = Concat(head_1, ..., head_h) · Wᴼ

Why? Different heads attend to different patterns:
  - Head 1: syntactic relationships
  - Head 2: semantic similarity
  - Head 3: positional patterns
```

## Transformer Architecture

```
         ┌──────────────────┐
         │   Output Probs   │
         │   (softmax)      │
         └────────┬─────────┘
         ┌────────┴─────────┐
         │   Linear         │
         └────────┬─────────┘
    ┌────────────────────────────┐
    │      Decoder Block × N     │
    │  ┌──────────────────────┐  │
    │  │ Masked Self-Attention│  │ ← Can only look at previous tokens
    │  │ Cross-Attention      │  │ ← Attend to encoder output
    │  │ Feed-Forward         │  │
    │  └──────────────────────┘  │
    └────────────┬───────────────┘
                 │
    ┌────────────────────────────┐
    │      Encoder Block × N     │
    │  ┌──────────────────────┐  │
    │  │ Self-Attention       │  │ ← Attend to all positions
    │  │ Feed-Forward         │  │
    │  │ (+ LayerNorm + Residual)│
    │  └──────────────────────┘  │
    └────────────┬───────────────┘
                 │
         ┌───────┴───────┐
         │  Positional   │
         │  Encoding     │
         └───────┬───────┘
         ┌───────┴───────┐
         │  Token        │
         │  Embeddings   │
         └───────────────┘
```

## Model Families

| Family | Architecture | Training | Examples |
|--------|-------------|----------|----------|
| **Encoder-only** | BERT | Masked LM (fill in blanks) | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | GPT | Causal LM (predict next token) | GPT-4, Claude, LLaMA, Gemini |
| **Encoder-Decoder** | T5 | Seq2seq | T5, BART, Flan-T5 |

## Key Concepts

### Positional Encoding
```
Sinusoidal (original):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Learned positions: most models now learn position embeddings
RoPE (Rotary): LLaMA, used for extrapolation to longer sequences
```

### Training LLMs

```
Pre-training → Fine-tuning → Alignment

1. Pre-training (Next token prediction on massive corpus)
   Loss = -Σ log P(xₜ | x₁...xₜ₋₁)
   Cost: millions of $ in compute

2. Supervised Fine-tuning (SFT)
   Train on (instruction, response) pairs

3. RLHF / DPO (Alignment)
   RLHF: Train reward model → PPO optimization
   DPO:  Direct preference optimization (simpler, no reward model)
```

### Inference Optimization

| Technique | What it Does |
|-----------|-------------|
| KV Cache | Cache key/value from previous tokens (avoid recompute) |
| Quantization | FP16 → INT8/INT4 (smaller, faster, slight quality loss) |
| Flash Attention | Memory-efficient attention (tiling, no materializing N×N matrix) |
| Speculative Decoding | Small model drafts, large model verifies (faster) |
| MoE | Mixture of Experts — only activate subset of parameters per token |

## Implement Self-Attention from Scratch

```python
import torch
import torch.nn.functional as F
import math

def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_model]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, seq, seq]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Example
batch, seq_len, d_model = 2, 10, 64
x = torch.randn(batch, seq_len, d_model)
output, attn_weights = self_attention(x, x, x)
print(output.shape)  # [2, 10, 64]
```

## Interview Questions

1. **Explain self-attention step by step. What are Q, K, V?**
2. **Why multi-head attention instead of single head?**
3. **BERT vs GPT — architecture and training differences?**
4. **Explain RLHF. Why is it needed after pre-training?**
5. **What is the KV cache? How does it speed up inference?**
6. **How does Flash Attention reduce memory?**
7. **Implement self-attention from scratch.**
8. **What is the difference between fine-tuning and RAG? When to use which?**
