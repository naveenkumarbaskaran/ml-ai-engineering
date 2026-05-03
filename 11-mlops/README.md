# MLOps & Production ML

## ML Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Data    │───→│  Train   │───→│  Deploy  │───→│ Monitor  │
│ Pipeline │    │ & Eval   │    │ & Serve  │    │ & Retrain│
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
     ↑                                               │
     └───────────────────────────────────────────────┘
                    Continuous Loop
```

## Model Serving Patterns

| Pattern | Latency | Use Case |
|---------|---------|----------|
| **REST API** | 50-200ms | General purpose, microservices |
| **Batch inference** | Minutes | Overnight scoring, recommendations |
| **Streaming** | 10-50ms | Real-time fraud detection, feeds |
| **Edge/On-device** | <10ms | Mobile (Core ML, TFLite) |

```python
# FastAPI model serving
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
async def predict(features: dict):
    X = preprocess(features)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
```

## Experiment Tracking

```python
import mlflow

mlflow.set_experiment("fraud_detection_v2")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 500)
    model.fit(X_train, y_train)
    mlflow.log_metric("auc", roc_auc_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
```

## Model Monitoring

### Data/Concept Drift
```
Data Drift:    P(X) changes — input distribution shifts
Concept Drift: P(Y|X) changes — relationship between features and target changes

Detection:
  - KL divergence between train and prod feature distributions
  - PSI (Population Stability Index)
  - Monitor prediction distribution over time
```

### Key Metrics to Monitor
- Prediction latency (p50, p99)
- Prediction distribution (shift from baseline)
- Feature distributions (drift detection)
- Business metrics (conversion, revenue)
- Data quality (nulls, schema violations)

## A/B Testing for ML

```
Deploy with shadow mode or canary:
1. Canary:  5% → 25% → 50% → 100% (gradual rollout)
2. Shadow:  New model runs in parallel, no impact
3. A/B:    Random assignment, measure business metric

Statistical significance:
  - Sample size calculation
  - Minimum 1-2 weeks
  - Watch for novelty/primacy effects
```

## Feature Store

```
Offline Features: Batch computed (daily), used for training
Online Features:  Low-latency (Redis/DynamoDB), used for serving

Tools: Feast, Tecton, AWS SageMaker Feature Store
```

## Interview Questions

1. **How do you deploy a model to production? Walk through the entire pipeline.**
2. **What is data drift? How do you detect and handle it?**
3. **How do you A/B test a new ML model?**
4. **What's the difference between online and offline feature stores?**
5. **How do you roll back a bad model deployment?**
6. **Design a CI/CD pipeline for an ML model.**
