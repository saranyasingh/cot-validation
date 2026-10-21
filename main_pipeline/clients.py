"""
LLM client abstractions for the CoT-validation pipeline.

Clients:
  openai    — OpenAI Responses API (gpt-5-nano), used for verification
  claude    — Anthropic Claude API (claude-opus-5-5 by default); 'anthropic' is an alias
  bedrock   — Claude via Amazon Bedrock
  kimi      — Moonshot/Kimi Chat Completions API, weaker model for reasoning
  deepseek  — DeepSeek Chat Completions API, weaker model for reasoning
  vllm      — locally-running vLLM server

Every `complete()` call records token usage and USD cost in an in-memory call log
(and, if `start_cost_log()` was called, appends it as a line to a JSONL file).
Use `call_log_mark()` + `summarize_costs(mark)` to get the cost of a span of calls.
"""

import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI as _OpenAI

load_dotenv()


# ─── Pricing ───────────────────────────────────────────────────────────────────

# USD per 1M tokens: (input, output). Exact model names are matched first, then the
# longest prefix, then the Claude family name (opus/sonnet/haiku) anywhere in the id.
# Models with no entry get cost=None (tokens are still logged).
PRICING = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),
}
CLAUDE_FAMILY_PRICING = {
    "opus": (5.00, 25.00),
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
}


def price_for(model: str):
    if model in PRICING:
        return PRICING[model]
    prefixes = [p for p in PRICING if model.startswith(p)]
    if prefixes:
        return PRICING[max(prefixes, key=len)]
    for family, price in CLAUDE_FAMILY_PRICING.items():
        if family in model:
            return price
    return None


# ─── Call log ──────────────────────────────────────────────────────────────────

_CALL_LOG: list[dict] = []
_cost_log_path = None
_cost_context: dict = {}


def start_cost_log(path: str):
    """Append every subsequent API call as a JSON line to `path`."""
    global _cost_log_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _cost_log_path = path


def set_cost_context(**fields):
    """Extra fields (e.g. item_id) attached to every subsequent call record."""
    _cost_context.clear()
    _cost_context.update(fields)


def call_log_mark() -> int:
    return len(_CALL_LOG)


def summarize_costs(start: int = 0, end: int = None) -> dict:
    calls = _CALL_LOG[start:end]
    by_model: dict = {}
    for c in calls:
        m = by_model.setdefault(c["model"], {
            "client": c["client"], "calls": 0,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        })
        m["calls"] += 1
        m["input_tokens"] += c["input_tokens"]
        m["output_tokens"] += c["output_tokens"]
        if c["cost_usd"] is None:
            m["cost_usd"] = None
        elif m["cost_usd"] is not None:
            m["cost_usd"] += c["cost_usd"]
    for m in by_model.values():
        if m["cost_usd"] is not None:
            m["cost_usd"] = round(m["cost_usd"], 6)
    known = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "total_cost_usd": round(sum(known), 6),
        "unpriced_calls": len(calls) - len(known),
        "by_model": by_model,
    }


def print_cost_summary(cost: dict):
    print(f"  -- API Cost --")
    print(f"  Calls                      : {cost['calls']}")
    print(f"  Tokens (in / out)          : {cost['input_tokens']:,} / {cost['output_tokens']:,}")
    print(f"  Total cost                 : ${cost['total_cost_usd']:.4f}")
    for model, m in cost["by_model"].items():
        usd = "unpriced" if m["cost_usd"] is None else f"${m['cost_usd']:.4f}"
        print(f"    {model:<36} {m['calls']:>4} calls  {usd}")
    if cost["unpriced_calls"]:
        print(f"  WARNING: {cost['unpriced_calls']} calls used models missing from PRICING in clients.py")


def _record_call(client_name, model, input_tokens, output_tokens, latency_s, price):
    cost = None
    if price is not None:
        cost = round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 8)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **_cost_context,
        "client": client_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "latency_s": round(latency_s, 3),
    }
    _CALL_LOG.append(record)
    if _cost_log_path:
        with open(_cost_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return record


# ─── Clients ───────────────────────────────────────────────────────────────────

class LLMClient:
    name = "base"
    model = ""

    def price(self):
        return price_for(self.model)

    def complete(self, prompt: str) -> str:
        start = time.monotonic()
        text, input_tokens, output_tokens = self._complete(prompt)
        _record_call(self.name, self.model, input_tokens or 0, output_tokens or 0,
                     time.monotonic() - start, self.price())
        return text

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""
        raise NotImplementedError


def _chat_usage(response):
    usage = response.usage
    if usage is None:
        return 0, 0
    return usage.prompt_tokens, usage.completion_tokens


class OpenAILLMClient(LLMClient):
    """Uses the OpenAI Responses API."""

    name = "openai"

    def __init__(self):
        self._client = _OpenAI()
        self.model = "gpt-5-nano"

    def _complete(self, prompt: str):
        response = self._client.responses.create(model=self.model, input=prompt)
        usage = response.usage
        return (
            response.output_text,
            usage.input_tokens if usage else 0,
            usage.output_tokens if usage else 0,
        )


class KimiLLMClient(LLMClient):
    """Uses the Moonshot/Kimi Chat Completions API (OpenAI-compatible)."""

    name = "kimi"

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=os.getenv("KIMI_API_KEY"),
        )
        self.model = os.getenv("KIMI_MODEL", "moonshot-v1-8k")

    def _complete(self, prompt: str):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content, *_chat_usage(response))


class DeepSeekLLMClient(LLMClient):
    """Uses the DeepSeek Chat Completions API (OpenAI-compatible)."""

    name = "deepseek"

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def _complete(self, prompt: str):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content, *_chat_usage(response))


class VLLMLLMClient(LLMClient):
    """Uses a locally-running vLLM server (OpenAI-compatible Chat Completions API)."""

    name = "vllm"

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="token-abc123",  # vLLM requires a non-empty key; value doesn't matter
        )
        self.model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    def price(self):
        return (0.0, 0.0)  # self-hosted

    def _complete(self, prompt: str):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content, *_chat_usage(response))


class ClaudeLLMClient(LLMClient):
    """Uses the Anthropic Messages API (Claude)."""

    name = "claude"

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))

    def _complete(self, prompt: str):
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            print(f"WARNING: {self.model} refused (stop_details={message.stop_details})")
        text = "".join(b.text for b in message.content if b.type == "text")
        return text, message.usage.input_tokens, message.usage.output_tokens


class BedrockLLMClient(ClaudeLLMClient):
    """Claude via Amazon Bedrock (Messages API endpoint). Auth: AWS_BEARER_TOKEN_BEDROCK
    (Bedrock API key) or standard AWS credentials (AWS_ACCESS_KEY_ID / profile)."""

    name = "bedrock"

    def __init__(self):
        from anthropic import AnthropicBedrockMantle
        self._client = AnthropicBedrockMantle(aws_region=os.getenv("AWS_REGION", "us-east-1"))
        self.model = os.getenv("BEDROCK_MODEL", "anthropic.claude-sonnet-5-5")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "64000"))


def make_client(name: str) -> LLMClient:
    if name == "openai":
        return OpenAILLMClient()
    if name in ("claude", "anthropic"):
        return ClaudeLLMClient()
    if name == "kimi":
        return KimiLLMClient()
    if name == "deepseek":
        return DeepSeekLLMClient()
    if name == "vllm":
        return VLLMLLMClient()
    if name == "bedrock":
        return BedrockLLMClient()
    raise ValueError(f"Unknown client '{name}'. Choose 'openai', 'claude', 'kimi', 'deepseek', 'vllm', 'anthropic', or 'bedrock'.")
