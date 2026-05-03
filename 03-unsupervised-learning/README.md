# Unsupervised Learning

## K-Means Clustering

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n = X.shape[0]
        # Random initialization
        idx = np.random.choice(n, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) for i in range(self.k)
            ])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self
```

### Choosing K
- **Elbow method**: Plot inertia vs K, look for "elbow"
- **Silhouette score**: Higher = better defined clusters

## PCA (Principal Component Analysis)

```
Goal: Reduce dimensions while preserving maximum variance

Steps:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Eigendecomposition → eigenvalues + eigenvectors
4. Project onto top-k eigenvectors
```

```python
class PCA:
    def fit_transform(self, X, n_components=2):
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:n_components]]
        self.explained_variance = eigenvalues[idx[:n_components]] / eigenvalues.sum()
        return X_centered @ self.components
```

## Anomaly Detection

| Method | How it Works | When to Use |
|--------|-------------|-------------|
| Isolation Forest | Isolates anomalies with fewer splits | Tabular, high-dim |
| One-Class SVM | Learns boundary around normal data | Small datasets |
| Autoencoder | High reconstruction error = anomaly | Complex patterns |
| Statistical (Z-score) | Points > 3σ from mean | Simple distributions |

## Interview Questions

1. **How does K-Means work? What are its limitations?**
2. **Explain PCA. When would you use it vs. t-SNE?**
3. **What's the difference between PCA and autoencoders for dimensionality reduction?**
4. **How do you detect anomalies in a dataset with no labels?**
5. **K-Means vs DBSCAN — when to use which?**
