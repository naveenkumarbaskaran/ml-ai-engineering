# Neural Networks & Deep Learning

## The Neuron

```
Inputs    Weights    Activation
x₁ ──w₁──┐
x₂ ──w₂──┤──→ Σ(wᵢxᵢ + b) ──→ σ(z) ──→ output
x₃ ──w₃──┘
```

## Backpropagation (The Heart of DL) ⭐

```
Forward Pass:  x → z = Wx + b → a = σ(z) → ... → Loss L

Backward Pass (Chain Rule):
  ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

Update:
  w = w - lr · ∂L/∂w
```

### Implement a Neural Network from Scratch

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):  # e.g., [784, 128, 64, 10]
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])  # He init
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self.z_values.append(z)
            a = self.softmax(z) if i == len(self.weights) - 1 else self.relu(z)
            self.activations.append(a)
        return a

    def backward(self, y_onehot, lr=0.001):
        m = y_onehot.shape[0]
        # Output layer gradient (softmax + cross-entropy)
        delta = self.activations[-1] - y_onehot

        for i in reversed(range(len(self.weights))):
            dw = (1/m) * self.activations[i].T @ delta
            db = (1/m) * delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_deriv(self.z_values[i-1])
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
```

## Activation Functions

| Function | Formula | Range | Best For |
|----------|---------|-------|----------|
| ReLU | max(0, z) | [0, ∞) | Hidden layers (default) |
| LeakyReLU | max(αz, z) | (-∞, ∞) | Fix dying ReLU |
| GELU | z·Φ(z) | (-0.17, ∞) | Transformers |
| Sigmoid | 1/(1+e⁻ᶻ) | (0, 1) | Binary output |
| Softmax | eᶻⁱ/Σeᶻʲ | (0, 1) | Multi-class output |
| Tanh | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | RNNs, bounded |

## Optimizers

```
SGD:        w = w - lr · ∇L                    (noisy, slow)
Momentum:   v = β·v + ∇L;  w = w - lr · v     (smoother)
Adam:       Combines momentum + RMSProp         (default choice ⭐)
            m = β₁·m + (1-β₁)·∇L              (1st moment)
            v = β₂·v + (1-β₂)·(∇L)²           (2nd moment)
            w = w - lr · m̂/(√v̂ + ε)
```

## Regularization in Deep Learning

| Technique | How | Why |
|-----------|-----|-----|
| Dropout | Randomly zero p% of neurons during training | Prevents co-adaptation |
| Batch Normalization | Normalize layer inputs: (x-μ)/σ · γ + β | Faster training, regularizes |
| Weight Decay (L2) | Add λ·‖w‖² to loss | Smaller weights |
| Early Stopping | Stop when validation loss increases | Prevent overfitting |
| Data Augmentation | Flip, rotate, crop images | More effective data |

## Interview Questions

1. **Walk through backpropagation step by step.**
2. **Why does ReLU work better than sigmoid in hidden layers?** (vanishing gradients)
3. **Explain batch normalization. Why does it help?**
4. **What is the dying ReLU problem? How to fix it?**
5. **Compare Adam vs SGD with momentum. When would you prefer SGD?**
6. **Implement a 2-layer neural network from scratch.**
7. **What is the difference between batch, mini-batch, and stochastic gradient descent?**
