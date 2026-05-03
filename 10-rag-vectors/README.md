# RAG & Vector Databases

## What is RAG?

```
RAG = Retrieval-Augmented Generation

Without RAG: LLM answers from training data (may hallucinate)
With RAG:    LLM answers using YOUR documents (grounded, current)

Pipeline:
  ┌──────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
  │  Query   │────→│  Retriever   │────→│  Context +  │────→│   LLM    │
  │          │     │ (Vector DB)  │     │   Query     │     │ Response │
  └──────────┘     └──────────────┘     └────────────┘     └──────────┘
```

## Indexing Pipeline

```
Documents → Chunk → Embed → Store in Vector DB

1. Load: PDF, HTML, Markdown, code, etc.
2. Chunk: Split into manageable pieces (512-1024 tokens)
3. Embed: Convert text → dense vector (e.g., OpenAI ada-002, Cohere)
4. Store: Save vectors + metadata in vector DB
```

### Chunking Strategies

| Strategy | Description | When |
|----------|-------------|------|
| Fixed size | Split every N tokens | Simple, generic |
| Recursive | Split by paragraph → sentence → word | Most common ⭐ |
| Semantic | Group by topic similarity | Best quality, slower |
| Document-aware | Respect headers, sections | Tech docs, legal |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
```

## Vector Databases

| Database | Type | Best For |
|----------|------|----------|
| **Pinecone** | Managed cloud | Production, easiest setup |
| **Weaviate** | Open source | Hybrid search, multi-modal |
| **Chroma** | Open source | Local dev, lightweight |
| **Qdrant** | Open source | Production, Rust-based, fast |
| **pgvector** | Postgres extension | Already using Postgres |
| **FAISS** | Library (Meta) | Offline, research, huge datasets |

## Search Strategies

### Dense Retrieval (Semantic)
```
query → embed → cosine similarity with stored vectors
Pros: Understands meaning ("car" matches "automobile")
Cons: May miss exact keywords
```

### Sparse Retrieval (BM25 / Keyword)
```
query → TF-IDF / BM25 scoring
Pros: Exact keyword matching
Cons: Misses synonyms
```

### Hybrid Search ⭐
```
score = α · dense_score + (1-α) · sparse_score
Best of both: semantic understanding + keyword precision
```

## Advanced RAG Techniques

| Technique | What | Why |
|-----------|------|-----|
| **Reranking** | Re-score top-K results with cross-encoder | Better relevance |
| **HyDE** | Generate hypothetical doc, then search | Better query embedding |
| **Multi-query** | LLM generates multiple query variants | Better recall |
| **Parent-child** | Retrieve small chunk, return parent document | Full context |
| **Contextual compression** | LLM extracts only relevant parts | Less noise |

## RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Data freshness | Real-time updates | Requires retraining |
| Cost | Vector DB + retrieval | GPU training cost |
| Hallucination | Grounded in sources | Can still hallucinate |
| Best for | Company docs, current data | Style, format, domain knowledge |

## Interview Questions

1. **Explain the RAG pipeline end-to-end.**
2. **How do you choose chunk size? What's the tradeoff?**
3. **Dense vs sparse retrieval — when to use which?**
4. **How do you evaluate RAG quality?** (context relevance, answer faithfulness, recall)
5. **RAG vs fine-tuning — when to use which?**
6. **Design a RAG system for a legal document search engine.**
