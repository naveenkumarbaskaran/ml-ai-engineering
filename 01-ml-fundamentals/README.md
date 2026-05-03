# ML Fundamentals

## The ML Landscape

```
                    Machine Learning
                    /       |        \
            Supervised  Unsupervised  Reinforcement
            /      \        |              |
     Classification Regression  Clustering  Policy/Value
```

## Bias-Variance Tradeoff ⭐

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias (Underfitting)          High Variance (Overfitting)
├── Model too simple              ├── Model too complex
├── Misses patterns               ├── Memorizes noise
├── Bad on train AND test         ├── Great on train, bad on test
└── Fix: more features,           └── Fix: more data, regularization,
    complex model                     dropout, simpler model
```

### The Sweet Spot
```
Error
│  \                    /
│   \    ___          /   ← Test error
│    \__/   \       /
│             \   /
│              \_/        ← Train error
│─────────────────────────→ Model Complexity
        Sweet Spot ↑
```

## Cross-Validation

```python
# K-Fold Cross-Validation (k=5)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified K-Fold (preserves class distribution)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## Evaluation Metrics

### Classification
| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Cost of false positive is high (spam) |
| Recall | TP/(TP+FN) | Cost of false negative is high (cancer) |
| F1 Score | 2·P·R/(P+R) | Imbalanced classes |
| AUC-ROC | Area under ROC | Ranking, threshold-independent |

### Regression
| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | mean((y - ŷ)²) | Penalizes large errors |
| MAE | mean(|y - ŷ|) | Robust to outliers |
| R² | 1 - SS_res/SS_tot | % variance explained |

## Regularization

```python
# L1 (Lasso) — Sparse features, feature selection
# Loss = MSE + λ Σ|wᵢ|
from sklearn.linear_model import Lasso

# L2 (Ridge) — Small weights, prevents overfitting
# Loss = MSE + λ Σwᵢ²
from sklearn.linear_model import Ridge

# ElasticNet — Combination of L1 + L2
from sklearn.linear_model import ElasticNet
```

## Interview Questions

1. **Explain bias-variance tradeoff with an example.**
2. **When would you choose precision over recall?**
3. **Why is accuracy misleading for imbalanced datasets?**
4. **What's the difference between L1 and L2 regularization?**
5. **Explain cross-validation. Why not just use train/test split?**
6. **How do you handle overfitting? Name 5 techniques.**
7. **What is the curse of dimensionality?**
8. **Explain the difference between parametric and non-parametric models.**
