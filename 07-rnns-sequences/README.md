# RNNs & Sequence Models

## Vanilla RNN

```
       h₁        h₂        h₃
       ↑         ↑         ↑
x₁ → [RNN] → [RNN] → [RNN] → ...
       ↓         ↓         ↓
       y₁        y₂        y₃

hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)
```

**Problem**: Vanishing gradients — can't learn long-range dependencies

## LSTM (Long Short-Term Memory) ⭐

```
Three gates + cell state:

┌─────────────────────────────────────────┐
│  fₜ = σ(Wf·[hₜ₋₁, xₜ])   Forget gate  │  → what to forget
│  iₜ = σ(Wi·[hₜ₋₁, xₜ])   Input gate   │  → what to store
│  oₜ = σ(Wo·[hₜ₋₁, xₜ])   Output gate  │  → what to output
│                                          │
│  c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ]) Candidate   │
│  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ  Cell state │  → "highway" for gradients
│  hₜ = oₜ ⊙ tanh(cₜ)        Hidden      │
└─────────────────────────────────────────┘
```

**Why LSTM solves vanishing gradients**: Cell state acts as a gradient highway — gradients flow through with minimal decay.

## GRU (Gated Recurrent Unit)

```
Simplified LSTM with 2 gates (fewer params):
  rₜ = σ(Wr·[hₜ₋₁, xₜ])    Reset gate
  zₜ = σ(Wz·[hₜ₋₁, xₜ])    Update gate
  h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
  hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

## Attention Mechanism (Bridge to Transformers) ⭐

```
Problem: Fixed-size hidden state bottleneck in encoder-decoder

Solution: Let decoder look at ALL encoder hidden states

           Encoder states: [h₁, h₂, h₃, h₄]
                              ↑   ↑   ↑   ↑
           Attention weights: 0.1 0.7 0.1 0.1  ← learned per decoder step
                              └───────────────→ Context vector (weighted sum)
```

```python
# Bahdanau attention (additive)
def attention(query, keys, values):
    # query: decoder hidden state [batch, d]
    # keys: encoder hidden states [batch, seq_len, d]
    scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # [batch, seq_len]
    weights = F.softmax(scores, dim=1)
    context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
    return context, weights
```

## Sequence-to-Sequence Tasks

| Task | Architecture | Example |
|------|-------------|---------|
| Classification | Many-to-one | Sentiment analysis |
| Translation | Many-to-many | English → French |
| Text generation | One-to-many | Prompt → story |
| Tagging | Many-to-many (sync) | NER, POS tagging |

## Interview Questions

1. **Explain the vanishing gradient problem in RNNs. How does LSTM solve it?**
2. **Walk through LSTM gates — what does each one do?**
3. **LSTM vs GRU — tradeoffs?** (GRU: fewer params, faster; LSTM: more expressive)
4. **Explain the attention mechanism in encoder-decoder models.**
5. **Why did Transformers largely replace RNNs?** (parallelization, long-range)
