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
        self.model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def make_client(name: str) -> LLMClient:
    if name == "openai":
        return OpenAILLMClient()
    if name == "kimi":
        return KimiLLMClient()
    if name == "deepseek":
        return DeepSeekLLMClient()
    if name == "vllm":
        return VLLMLLMClient()
    raise ValueError(f"Unknown client '{name}'. Choose 'openai', 'kimi', 'deepseek', or 'vllm'.")
