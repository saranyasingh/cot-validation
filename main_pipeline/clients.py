"""
LLM client abstractions for the CoT-validation pipeline.

Clients:
  openai    — OpenAI Responses API (gpt-5-nano), used for verification
  kimi      — Moonshot/Kimi Chat Completions API, weaker model for reasoning
  deepseek  — DeepSeek Chat Completions API, weaker model for reasoning
"""

import os
from dotenv import load_dotenv
from openai import OpenAI as _OpenAI

load_dotenv()


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """Uses the OpenAI Responses API."""

    def __init__(self):
        self._client = _OpenAI()
        self.model = "gpt-5-nano"

    def complete(self, prompt: str) -> str:
        response = self._client.responses.create(model=self.model, input=prompt)
        return response.output_text


class KimiLLMClient(LLMClient):
    """Uses the Moonshot/Kimi Chat Completions API (OpenAI-compatible)."""

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=os.getenv("KIMI_API_KEY"),
        )
        self.model = os.getenv("KIMI_MODEL", "moonshot-v1-8k")

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class DeepSeekLLMClient(LLMClient):
    """Uses the DeepSeek Chat Completions API (OpenAI-compatible)."""

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class VLLMLLMClient(LLMClient):
    """Uses a locally-running vLLM server (OpenAI-compatible Chat Completions API)."""

    def __init__(self):
        self._client = _OpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="token-abc123",  # vLLM requires a non-empty key; value doesn't matter
        )
        self.model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class AnthropicLLMClient(LLMClient):
    """Uses the Anthropic Messages API (Claude)."""

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))

    def complete(self, prompt: str) -> str:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            print(f"WARNING: {self.model} refused (stop_details={message.stop_details})")
        return "".join(b.text for b in message.content if b.type == "text")


class BedrockLLMClient(AnthropicLLMClient):
    """Claude via Amazon Bedrock (Messages API endpoint). Auth: AWS_BEARER_TOKEN_BEDROCK
    (Bedrock API key) or standard AWS credentials (AWS_ACCESS_KEY_ID / profile)."""

    def __init__(self):
        from anthropic import AnthropicBedrockMantle
        self._client = AnthropicBedrockMantle(aws_region=os.getenv("AWS_REGION", "us-east-1"))
        self.model = os.getenv("BEDROCK_MODEL", "anthropic.claude-fable-5-1")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "64000"))


def make_client(name: str) -> LLMClient:
    if name == "openai":
        return OpenAILLMClient()
    if name == "kimi":
        return KimiLLMClient()
    if name == "deepseek":
        return DeepSeekLLMClient()
    if name == "vllm":
        return VLLMLLMClient()
    if name == "anthropic":
        return AnthropicLLMClient()
    if name == "bedrock":
        return BedrockLLMClient()
    raise ValueError(f"Unknown client '{name}'. Choose 'openai', 'kimi', 'deepseek', 'vllm', 'anthropic', or 'bedrock'.")
