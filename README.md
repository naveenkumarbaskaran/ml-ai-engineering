# 🧠 ML & AI Engineering — MAANG Interview Mastery

<div align="center">

![ML](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![DL](https://img.shields.io/badge/Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![AI](https://img.shields.io/badge/Agentic_AI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LLM](https://img.shields.io/badge/LLMs-000000?style=for-the-badge&logo=anthropic&logoColor=white)

**From gradient descent to autonomous agents — everything you need for ML/AI roles at top tech companies.**

*🤖 Interactive AI Tutor Built-In — Ask Claude, Copilot, or any LLM to quiz you, explain concepts, or generate practice problems right inside your editor.*

</div>

---

## 🗺️ Learning Path

### Phase 1: ML Foundations (Weeks 1-3)
| Topic | File | Key Concepts |
|-------|------|-------------|
| ML Fundamentals | [01-ml-fundamentals/](01-ml-fundamentals/) | Bias-variance, overfitting, cross-validation |
| Supervised Learning | [02-supervised-learning/](02-supervised-learning/) | Linear/Logistic Regression, SVM, Decision Trees, Ensemble Methods |
| Unsupervised Learning | [03-unsupervised-learning/](03-unsupervised-learning/) | K-Means, PCA, DBSCAN, Anomaly Detection |
| Feature Engineering | [04-feature-engineering/](04-feature-engineering/) | Encoding, scaling, selection, embeddings |

### Phase 2: Deep Learning (Weeks 4-6)
| Topic | File | Key Concepts |
|-------|------|-------------|
| Neural Networks | [05-neural-networks/](05-neural-networks/) | Backprop, activations, optimizers, regularization |
| CNNs | [06-cnns/](06-cnns/) | Convolutions, architectures (ResNet, EfficientNet) |
| RNNs & Sequence Models | [07-rnns-sequences/](07-rnns-sequences/) | LSTM, GRU, attention mechanism |
| Transformers & LLMs | [08-transformers-llms/](08-transformers-llms/) | Self-attention, BERT, GPT, fine-tuning, RLHF |

### Phase 3: Applied AI & Agentic Systems (Weeks 7-8)
| Topic | File | Key Concepts |
|-------|------|-------------|
| Agentic AI Patterns | [09-agentic-ai/](09-agentic-ai/) | ReAct, tool use, planning, multi-agent, memory |
| RAG & Vector Databases | [10-rag-vectors/](10-rag-vectors/) | Embeddings, chunking, retrieval, hybrid search |
| MLOps & Production | [11-mlops/](11-mlops/) | Model serving, monitoring, drift, A/B testing |
| ML System Design | [12-ml-system-design/](12-ml-system-design/) | Recommendation systems, search ranking, fraud detection |

---

## 🤖 AI-Powered Study Mode

This repo has built-in AI tutor support for **6 LLM providers**:

| Provider | Setup |
|----------|-------|
| **Claude** (Anthropic) | Open in Claude Code — reads `CLAUDE.md` automatically |
| **GitHub Copilot** | Open in VS Code — reads `.github/copilot-instructions.md` |
| **OpenAI / GPT-4** | Use `.prompts/` files or `tutor.py --provider openai` |
| **Google Gemini** | `tutor.py --provider gemini` |
| **Ollama (Local)** | `tutor.py --provider ollama --model llama3` |
| **Azure OpenAI** | `tutor.py --provider azure` |

### Quick Start
```bash
# Install tutor dependencies
pip install -r ai-tutor/requirements.txt

# Interactive quiz mode
python ai-tutor/tutor.py quiz --topic transformers

# Explain a concept
python ai-tutor/tutor.py explain "attention mechanism"

# Mock interview
python ai-tutor/tutor.py interview --role "ML Engineer" --company google
```

### In-Editor (Claude/Copilot)
Just ask in chat:
- *"Quiz me on backpropagation"*
- *"Explain batch normalization like I'm 5"*
- *"Give me an ML system design for a recommendation engine"*
- *"What's the difference between RAG and fine-tuning?"*

---

## 📊 Interview Question Bank

| Category | Count | Difficulty |
|----------|-------|-----------|
| ML Theory | 40 | Medium-Hard |
| Deep Learning | 35 | Hard |
| Coding (implement from scratch) | 25 | Medium-Hard |
| ML System Design | 15 | Hard |
| Agentic AI / LLM | 20 | Medium-Hard |
| **Total** | **135** | |

---

## 🎯 Company-Specific Focus

| Company | Focus Areas |
|---------|------------|
| **Google** | ML fundamentals, TensorFlow, large-scale systems, research depth |
| **Meta** | Recommendation systems, ranking, PyTorch, A/B testing |
| **Amazon** | Applied ML, personalization, system design, SageMaker |
| **Apple** | On-device ML, privacy-preserving ML, Core ML |
| **Microsoft** | Azure ML, responsible AI, LLM applications |
| **OpenAI/Anthropic** | Transformers, RLHF, alignment, agentic patterns |

---

> **How to use this repo**: Work through phases sequentially. Each topic has theory notes, implementation exercises, and interview questions. Use the AI tutor for adaptive learning — it adjusts difficulty based on your responses.
