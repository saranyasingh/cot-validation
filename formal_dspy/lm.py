"""Run DSPy through the repo's own LLM clients, so auth, vLLM/Bedrock setup and cost logging stay the same."""

import os
import sys

import dspy
from dspy.clients.engines.base import validate_request
from dspy.lm15 import Message, Response, TextPart, Usage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# main_pipeline/main.py imports its siblings by bare name (import verify, import tptp)
sys.path.insert(0, os.path.join(REPO_ROOT, "main_pipeline"))

from main_pipeline.clients import make_client  # noqa: E402

CLIENTS = ["openai", "claude", "kimi", "deepseek", "vllm", "bedrock"]


def _text(parts) -> str:
    if isinstance(parts, str):
        return parts
    return "".join(p.text for p in parts if isinstance(p, TextPart))


class ClientEngine:
    """A DSPy engine that sends the rendered request to an LLMClient as one prompt."""

    def __init__(self, client):
        self.client = client

    def complete(self, request):
        validate_request(request)
        blocks = [_text(request.system)] if request.system is not None else []
        for m in request.messages:
            blocks.append(_text(m.parts) if m.role == "user" else f"[{m.role}]\n{_text(m.parts)}")
        output = self.client.complete("\n\n".join(blocks))
        return Response(id=None, model=self.client.model, message=Message.assistant([TextPart(output)]),
                        finish_reason="stop",
                        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))


def make_lm(name: str, model: str | None = None) -> dspy.BaseLM:
    """A DSPy LM backed by make_client(name). `model` overrides the client's default model."""
    from dotenv import load_dotenv
    load_dotenv()
    if name == "bedrock" and os.getenv("BEDROCK_API_KEY") and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["BEDROCK_API_KEY"]
    client = make_client(name)
    if model:
        client.model = model
    return dspy.BaseLM(model=client.model, engine=ClientEngine(client))
