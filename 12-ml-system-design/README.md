# ML System Design

## Framework for ML System Design Interviews

```
1. Clarify Requirements (2 min)
   - What is the business goal?
   - What metrics matter? (business + ML)
   - Scale? Latency requirements?

2. Data (5 min)
   - What data is available?
   - How to collect labels?
   - Feature engineering

3. Model (5 min)
   - Baseline → Simple → Complex
   - Training pipeline
   - Offline evaluation

4. Serving (5 min)
   - Online vs batch inference
   - Feature serving
   - Caching, fallbacks

5. Monitoring & Iteration (3 min)
   - A/B testing
   - Drift detection
   - Feedback loops
```

---

## Case Study: Recommendation System (Netflix/YouTube)

### Requirements
- Recommend videos from catalog of 1M+ items
- Personalized per user
- Latency: < 200ms
- Update as user behavior changes

### Architecture

```
  ┌──────────────────────────────────────────────┐
  │              Recommendation Pipeline          │
  │                                               │
  │  Candidate Generation (1M → 1000)            │
  │    ├── Collaborative Filtering (user-user, item-item)
  │    ├── Content-based (metadata similarity)    │
  │    └── Two-tower model (user & item embeddings)│
  │                                               │
  │  Ranking (1000 → 50)                         │
  │    ├── Features: user history, item features, context
  │    ├── Model: LightGBM or DNN               │
  │    └── Optimize for engagement (CTR, watch time)
  │                                               │
  │  Re-ranking (business rules)                  │
  │    ├── Diversity (don't show 5 action movies) │
  │    ├── Freshness (boost new content)          │
  │    └── Policy (remove inappropriate)          │
  └──────────────────────────────────────────────┘
```

### Two-Tower Model
```
User Tower:                Item Tower:
  user_id                    item_id
  watch_history              genre, tags
  demographics               description_embedding
       ↓                         ↓
    [DNN]                     [DNN]
       ↓                         ↓
  user_embedding            item_embedding
       └────── dot product ──────┘
              = relevance score
```

---

## Case Study: Search Ranking (Google)

### Multi-Stage Pipeline
```
Query → Retrieval (BM25 + embedding) → 1000 docs
     → Pre-ranking (lightweight model) → 100 docs
     → Ranking (complex DNN) → 10 docs
     → Re-ranking (freshness, diversity, ads)
```

### Features
```
Query features:  length, intent (navigational/informational), language
Document features: PageRank, freshness, quality score, length
Cross features:  BM25 score, embedding similarity, click-through rate
User features:   location, search history, device
```

---

## Case Study: Fraud Detection

```
Requirements:
  - Real-time (< 100ms per transaction)
  - High recall (catch fraud) with acceptable precision
  - Handle class imbalance (0.1% fraud)

Architecture:
  Transaction → Feature Extraction → [Rule Engine + ML Model] → Decision
                                           ↓
                                     Human Review Queue
                                     (borderline cases)

Handling Imbalance:
  - SMOTE / oversampling minority class
  - Class weights (cost-sensitive learning)
  - Anomaly detection as first filter
  - Ensemble: rules for known patterns + ML for novel fraud
```

---

## Interview Questions

1. **Design a recommendation system for YouTube.** (candidate gen → ranking → re-ranking)
2. **Design a search autocomplete system.** (trie + ML ranking + personalization)
3. **Design a spam/fraud detection system.** (real-time, imbalanced, feedback loop)
4. **Design an ad click prediction system.** (CTR prediction, feature store, A/B testing)
5. **Design a content moderation system.** (multi-modal: text + image + video)
6. **How would you build a personalized news feed?** (similar to Twitter/Facebook)
