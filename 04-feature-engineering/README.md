# Feature Engineering

## The Feature Engineering Pipeline

```
Raw Data → Clean → Transform → Encode → Select → Scale → Model
```

## Handling Missing Data

```python
# Strategy depends on % missing and pattern
import pandas as pd

# Simple imputation
df['col'].fillna(df['col'].median(), inplace=True)  # numeric
df['col'].fillna(df['col'].mode()[0], inplace=True)  # categorical

# KNN Imputer (uses similar rows)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Create binary indicator for missingness (can be a feature!)
df['col_missing'] = df['col'].isna().astype(int)
```

## Encoding Categorical Variables

| Method | When | Example |
|--------|------|---------|
| Label Encoding | Ordinal (low/med/high) | `LabelEncoder()` |
| One-Hot Encoding | Nominal, few categories | `pd.get_dummies()` |
| Target Encoding | High cardinality, tree models | mean(target) per category |
| Embedding | Very high cardinality | Neural network embedding layer |

## Feature Scaling

```python
# StandardScaler: zero mean, unit variance (most common)
# Use when: algorithm uses distances (SVM, KNN, PCA)
from sklearn.preprocessing import StandardScaler

# MinMaxScaler: scale to [0, 1]
# Use when: neural networks, bounded range needed
from sklearn.preprocessing import MinMaxScaler

# RobustScaler: uses median/IQR (robust to outliers)
from sklearn.preprocessing import RobustScaler
```

## Feature Selection

```python
# 1. Filter methods (fast, univariate)
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y)

# 2. Wrapper methods (slow, greedy)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=10)

# 3. Embedded methods (built into model)
importances = xgb_model.feature_importances_

# 4. L1 regularization (automatic feature selection)
from sklearn.linear_model import Lasso
```

## Interview Questions

1. **How do you handle a categorical feature with 10,000 unique values?**
2. **When is feature scaling necessary? Which algorithms need it?**
3. **Explain target encoding. What's the risk?** (data leakage)
4. **How do you handle missing data? When would you drop rows vs. impute?**
5. **What's the difference between filter, wrapper, and embedded feature selection?**
