# Agentic AI Patterns

## What Makes an Agent?

```
Traditional LLM:  Input → Model → Output (one-shot)

Agentic LLM:      Input → [Think → Act → Observe] → ... → Output
                          └─── Loop until done ────┘
```

An **AI Agent** = LLM + Tools + Memory + Planning

---

## Core Patterns

### 1. ReAct (Reasoning + Acting) ⭐

```
Loop:
  Thought: "I need to find the current stock price of AAPL"
  Action:  search_stock("AAPL")
  Observation: "AAPL: $198.50"

  Thought: "Now I need the P/E ratio to assess valuation"
  Action:  get_financials("AAPL")
  Observation: "P/E: 31.2, Revenue: $385B"

  Thought: "I have enough info to answer"
  Answer:  "AAPL is trading at $198.50 with a P/E of 31.2..."
```

### 2. Tool Use / Function Calling

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    },
    {
        "name": "run_python",
        "description": "Execute Python code",
        "parameters": {
            "code": {"type": "string"}
        }
    }
]

# LLM decides WHICH tool to call and with WHAT parameters
# The framework executes the tool and feeds result back to LLM
```

### 3. Planning (Task Decomposition)

```
User: "Create a market analysis report for Tesla"

Agent Plan:
  1. Research Tesla's recent financial performance
  2. Analyze competitor landscape
  3. Gather market sentiment data
  4. Assess regulatory environment
  5. Generate report with visualizations
  6. Review and refine

Each step may involve multiple tool calls and sub-reasoning
```

### 4. Multi-Agent Systems

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Researcher  │────→│   Writer     │────→│   Reviewer   │
│  Agent       │     │   Agent      │     │   Agent      │
│  (searches)  │     │  (drafts)    │     │  (critiques) │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                          ┌───────▼───────┐
                                          │  Final Output  │
                                          └────────────────┘
```

**Frameworks**: CrewAI, AutoGen, LangGraph, Swarm

### 5. Memory Systems

```
Short-term Memory:  Conversation context (in-context window)
Long-term Memory:   Vector database (persist across sessions)
Episodic Memory:    Past interactions indexed by similarity
Working Memory:     Scratchpad for current task state

Implementation:
  Store: Embed experience → save to vector DB
  Recall: Embed query → similarity search → inject into prompt
```

---

## Frameworks Comparison

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **LangGraph** | Complex workflows, cycles | Graph-based state machines |
| **LangChain** | Chains of LLM calls | Large ecosystem, integrations |
| **CrewAI** | Multi-agent collaboration | Role-based agent teams |
| **AutoGen** | Agent conversations | Microsoft, multi-agent chat |
| **Semantic Kernel** | Enterprise .NET/Python | Microsoft, planner + plugins |
| **DSPy** | Prompt optimization | Auto-compiles prompts |

---

## LangGraph Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    next_action: str

def researcher(state: AgentState) -> AgentState:
    # Use LLM to research
    result = llm.invoke(state["messages"] + ["Research the topic..."])
    return {"messages": state["messages"] + [result], "next_action": "write"}

def writer(state: AgentState) -> AgentState:
    result = llm.invoke(state["messages"] + ["Write a report based on research..."])
    return {"messages": state["messages"] + [result], "next_action": "review"}

def router(state: AgentState) -> str:
    return state["next_action"]

# Build graph
graph = StateGraph(AgentState)
graph.add_node("research", researcher)
graph.add_node("write", writer)
graph.add_conditional_edges("research", router, {"write": "write"})
graph.add_edge("write", END)
graph.set_entry_point("research")

app = graph.compile()
```

---

## Interview Questions

1. **What is an AI agent? How is it different from a chatbot?**
2. **Explain the ReAct pattern. Why is it effective?**
3. **How do you implement tool use / function calling?**
4. **What are the challenges of multi-agent systems?** (coordination, loops, cost)
5. **How would you add long-term memory to an agent?**
6. **Design an agent system for [customer support / code review / data analysis].**
7. **LangChain vs LangGraph — when to use which?**
8. **How do you evaluate agent performance?** (task completion, cost, latency)
