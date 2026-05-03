# CNNs — Convolutional Neural Networks

## Core Concept

```
Image → [Conv → ReLU → Pool] × N → Flatten → FC → Output

Conv Layer:
┌─────────────────┐     ┌───┐
│  Input Image     │  *  │ K │  =  Feature Map
│  (H × W × C)    │     │   │     (detects edges, textures, etc.)
└─────────────────┘     └───┘
                        (3×3 kernel)
```

## Key Operations

### Convolution
```
Output size = (Input - Kernel + 2·Padding) / Stride + 1

Example: Input=32×32, Kernel=3×3, Padding=1, Stride=1
  → Output = (32 - 3 + 2) / 1 + 1 = 32×32 (same size)
```

### Pooling
```
Max Pool (2×2, stride 2): Halves spatial dimensions
┌──┬──┐
│ 1│ 3│ → 4 (takes max)
│ 2│ 4│
└──┴──┘
```

## Architectures

| Model | Year | Key Innovation | Params |
|-------|------|---------------|--------|
| LeNet | 1998 | First CNN | 60K |
| AlexNet | 2012 | ReLU, dropout, GPU | 60M |
| VGG-16 | 2014 | Stacked 3×3 convs | 138M |
| GoogLeNet | 2014 | Inception module (multi-scale) | 6.8M |
| ResNet | 2015 | Skip connections ⭐ | 25M |
| EfficientNet | 2019 | Compound scaling | 5-66M |

### ResNet Skip Connection ⭐
```
x → [Conv → BN → ReLU → Conv → BN] → (+x) → ReLU
     └───────────────────────────────────┘
            "Skip / Residual connection"

Why it works:
- Solves vanishing gradient in deep networks
- Network can learn identity (skip = do nothing)
- Enables training 100+ layer networks
```

## Transfer Learning

```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

## Interview Questions

1. **Explain how a convolutional layer works. Why convolutions over fully connected?**
2. **What is a residual connection and why does it work?**
3. **Calculate the output size of a conv layer.** (formula)
4. **How does transfer learning work? When would you fine-tune vs. feature extract?**
5. **What is 1×1 convolution and why is it useful?** (channel reduction, GoogLeNet)
