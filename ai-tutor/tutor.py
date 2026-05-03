"""
AI Tutor — Multi-Provider LLM Interactive Study Tool
=====================================================
Supports: OpenAI, Anthropic (Claude), Google Gemini, Ollama (local), Azure OpenAI, AWS Bedrock

Usage:
  python tutor.py quiz --topic "binary search"
  python tutor.py explain "backpropagation"
  python tutor.py interview --role "Software Engineer" --company google
  python tutor.py chat  (free-form conversation)

Provider selection:
  python tutor.py quiz --topic "trees" --provider openai
  python tutor.py quiz --topic "trees" --provider anthropic
  python tutor.py quiz --topic "trees" --provider gemini
  python tutor.py quiz --topic "trees" --provider ollama --model llama3
  python tutor.py quiz --topic "trees" --provider azure
  python tutor.py quiz --topic "trees" --provider bedrock

Environment variables (set the ones you need):
  OPENAI_API_KEY        — OpenAI
  ANTHROPIC_API_KEY     — Anthropic (Claude)
  GOOGLE_API_KEY        — Google Gemini
  OLLAMA_BASE_URL       — Ollama (default: http://localhost:11434)
  AZURE_OPENAI_ENDPOINT — Azure OpenAI
  AZURE_OPENAI_KEY      — Azure OpenAI
  AZURE_OPENAI_DEPLOYMENT — Azure deployment name
  AWS_REGION            — AWS Bedrock region
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod


# ─── Provider Abstraction ───

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send messages and return response text."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def chat(self, messages):
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content

    def name(self):
        return f"OpenAI ({self.model})"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def chat(self, messages):
        # Extract system message if present
        system = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=user_messages,
        )
        return resp.content[0].text

    def name(self):
        return f"Anthropic ({self.model})"


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def chat(self, messages):
        # Convert to Gemini format
        history = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            history.append({"role": role, "parts": [m["content"]]})
        chat = self.model.start_chat(history=history[:-1])
        resp = chat.send_message(history[-1]["parts"][0])
        return resp.text

    def name(self):
        return f"Google Gemini ({self.model_name})"


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3"):
        import ollama
        self._client = ollama
        self.model = model
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def chat(self, messages):
        resp = self._client.chat(model=self.model, messages=messages)
        return resp["message"]["content"]

    def name(self):
        return f"Ollama ({self.model})"


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, model: str = None):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-02-15-preview",
        )
        self.model = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    def chat(self, messages):
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content

    def name(self):
        return f"Azure OpenAI ({self.model})"


class BedrockProvider(LLMProvider):
    def __init__(self, model: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        import boto3, json
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.model = model
        self._json = json

    def chat(self, messages):
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system,
            "messages": user_msgs,
        }
        resp = self.bedrock.invoke_model(modelId=self.model, body=self._json.dumps(body))
        result = self._json.loads(resp["body"].read())
        return result["content"][0]["text"]

    def name(self):
        return f"AWS Bedrock ({self.model})"


# ─── Provider Factory ───

def get_provider(provider_name: str, model: str = None) -> LLMProvider:
    providers = {
        "openai": lambda: OpenAIProvider(model or "gpt-4o"),
        "anthropic": lambda: AnthropicProvider(model or "claude-sonnet-4-20250514"),
        "gemini": lambda: GeminiProvider(model or "gemini-2.0-flash"),
        "ollama": lambda: OllamaProvider(model or "llama3"),
        "azure": lambda: AzureOpenAIProvider(model),
        "bedrock": lambda: BedrockProvider(model or "anthropic.claude-3-sonnet-20240229-v1:0"),
    }
    if provider_name not in providers:
        print(f"❌ Unknown provider: {provider_name}")
        print(f"   Available: {', '.join(providers.keys())}")
        sys.exit(1)
    try:
        return providers[provider_name]()
    except KeyError as e:
        print(f"❌ Missing environment variable for {provider_name}: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Missing package for {provider_name}: {e}")
        print(f"   Run: pip install -r ai-tutor/requirements.txt")
        sys.exit(1)


def auto_detect_provider() -> str:
    """Auto-detect which provider to use based on available env vars."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("AZURE_OPENAI_KEY"):
        return "azure"
    # Try Ollama (local, no key needed)
    try:
        import ollama
        return "ollama"
    except ImportError:
        pass
    print("❌ No LLM provider configured.")
    print("   Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
    print("   Or install ollama for local models.")
    sys.exit(1)


# ─── System Prompts ───

def get_system_prompt(mode: str, repo_context: str = "") -> str:
    base = f"""You are an expert AI tutor for technical interview preparation.
You are knowledgeable, encouraging, and rigorous.
{repo_context}"""

    if mode == "quiz":
        return base + """
When quizzing:
1. Present ONE clear problem with examples
2. Wait for the user's answer
3. Analyze correctness, completeness, and efficiency
4. Provide the optimal answer if they missed something
5. Rate difficulty and suggest next topic"""

    elif mode == "explain":
        return base + """
When explaining:
1. Start with a simple analogy
2. Give the technical explanation with visuals (ASCII diagrams)
3. Show code implementation
4. Provide practice problems to solidify understanding"""

    elif mode == "interview":
        return base + """
When conducting a mock interview:
1. Act as a professional interviewer
2. Present problems progressively (easy → hard)
3. Give hints if the candidate is stuck
4. Ask follow-up questions
5. Give structured feedback at the end with scores"""

    else:  # chat
        return base + "\nHelp the user with any questions about the topics in this repository."


# ─── Interactive Loop ───

def interactive_session(provider: LLMProvider, system_prompt: str, initial_message: str = None):
    print(f"\n🤖 AI Tutor — powered by {provider.name()}")
    print("   Type 'quit' or 'exit' to end the session\n")

    messages = [{"role": "system", "content": system_prompt}]

    if initial_message:
        messages.append({"role": "user", "content": initial_message})
        print(f"📝 You: {initial_message}\n")
        response = provider.chat(messages)
        print(f"🤖 Tutor: {response}\n")
        messages.append({"role": "assistant", "content": response})

    while True:
        try:
            user_input = input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Session ended.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Session ended. Good luck with your prep!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        response = provider.chat(messages)
        print(f"\n🤖 Tutor: {response}\n")
        messages.append({"role": "assistant", "content": response})


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description="AI Tutor — Multi-Provider LLM Study Tool")
    parser.add_argument("mode", choices=["quiz", "explain", "interview", "chat"],
                        help="Interaction mode")
    parser.add_argument("topic", nargs="?", default=None,
                        help="Topic to study (for quiz/explain modes)")
    parser.add_argument("--provider", "-p", default=None,
                        help="LLM provider: openai, anthropic, gemini, ollama, azure, bedrock")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (provider-specific)")
    parser.add_argument("--role", default="Software Engineer",
                        help="Interview role (for interview mode)")
    parser.add_argument("--company", default="google",
                        help="Target company (for interview mode)")
    args = parser.parse_args()

    # Resolve provider
    provider_name = args.provider or auto_detect_provider()
    provider = get_provider(provider_name, args.model)

    # Build system prompt
    system_prompt = get_system_prompt(args.mode)

    # Build initial message
    if args.mode == "quiz" and args.topic:
        initial_msg = f"Quiz me on {args.topic} with an interview-level question."
    elif args.mode == "explain" and args.topic:
        initial_msg = f"Explain {args.topic} in depth with examples and code."
    elif args.mode == "interview":
        initial_msg = f"Run a mock {args.company}-style interview for a {args.role} position."
    else:
        initial_msg = None

    interactive_session(provider, system_prompt, initial_msg)


if __name__ == "__main__":
    main()
