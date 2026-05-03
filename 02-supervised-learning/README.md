# Supervised Learning

## Algorithm Cheatsheet

```
           Need            → Try This
           ─────────────────────────────
           Linear relationship?   → Linear/Ridge Regression
           Binary classification? → Logistic Regression, SVM
           Non-linear, tabular?   → XGBoost / Random Forest ⭐
           Small dataset?         → SVM with kernel
           Need interpretability? → Decision Tree, Logistic Reg
           Large dataset?         → Neural Network
```

## Linear Regression

```python
# From scratch
class LinearRegression:
    def fit(self, X, y, lr=0.01, epochs=1000):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            # Gradients
            dw = (1/n) * X.T @ (y_pred - y)
            db = (1/n) * np.sum(y_pred - y)
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X):
        return X @ self.w + self.b
```

## Logistic Regression

```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))

Loss: Binary Cross-Entropy
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

```python
class LogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, lr=0.01, epochs=1000):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(epochs):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            dw = (1/n) * X.T @ (y_pred - y)
            db = (1/n) * np.sum(y_pred - y)
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X, threshold=0.5):
        return (self.sigmoid(X @ self.w + self.b) >= threshold).astype(int)
```

## Decision Trees & Ensemble Methods

### Random Forest
```
Random Forest = Bagging + Feature Randomness
├── Train N trees on bootstrap samples
├── Each split considers random √d features
├── Predict by majority vote (classification) or average (regression)
└── Reduces variance (less overfitting than single tree)
```

### Gradient Boosting (XGBoost) ⭐
```
Boosting = Sequential error correction
├── Train tree 1 → compute residuals
├── Train tree 2 on residuals → compute new residuals
├── Train tree 3 on new residuals → ...
└── Final = weighted sum of all trees

Key hyperparameters:
  n_estimators: 100-1000
  learning_rate: 0.01-0.3
  max_depth: 3-10
  subsample: 0.7-1.0
```

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
```

## SVM (Support Vector Machine)

```
Maximize margin between classes
Kernel trick: map to higher dimension without computing it

Kernels:
  Linear:     K(x,y) = xᵀy
  Polynomial: K(x,y) = (xᵀy + c)^d
  RBF (Gaussian): K(x,y) = exp(-γ||x-y||²)  ← most common
```

## Interview Questions

1. **Why does Random Forest reduce variance but not bias?**
2. **Explain the kernel trick in SVM.**
3. **When would XGBoost outperform a neural network?**
4. **Implement logistic regression from scratch with gradient descent.**
5. **Difference between bagging and boosting?**
6. **How does a decision tree decide where to split?** (Gini, entropy, information gain)
