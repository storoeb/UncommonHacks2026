"""FastAPI agent — OpenAI-compatible `/v1/chat/completions` endpoint.

The Prophet Arena harness POSTs a standard OpenAI chat request. We:
  1. Pull the last user message's content.
  2. Run the full forecasting pipeline (parse → ensemble → retrieval →
     calibrator stub → α-shading stub).
  3. Return an OpenAI-format response with a `{"probabilities": [...]}`
     JSON block in `choices[0].message.content` so the evaluator can parse it.

Non-standard top-level field `prophet_debug` carries our intermediate values
for the live demo (judges' parsers ignore unknown top-level keys).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from prophet_agent.pipeline import PipelineResult, run_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="ProphetHacks 2026 Agent", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / response models — OpenAI-compatible subset.
# ---------------------------------------------------------------------------

class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    """Permissive subset of the OpenAI chat-completion schema."""

    model_config = ConfigDict(extra="allow")
    model: str = "prophet-agent"
    messages: list[_ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _extract_user_content(messages: list[_ChatMessage]) -> str:
    """Return the last user message content (or last message content as fallback)."""
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    # Fallback: most recent message with content, whatever its role.
    for m in reversed(messages):
        if m.content:
            return m.content
    return ""


def _format_message_content(result: PipelineResult) -> str:
    """Compose the chat message body the evaluator parses."""
    probs = [round(p, 6) for p in result.p_final]
    payload = {"probabilities": probs}
    outcomes = ", ".join(result.parsed.outcomes)
    explanation = (
        f"Based on ensemble forecast over {len(result.ensemble_raw)} models "
        f"(disagreement variance {result.p_ensemble_var:.4f}) and historical "
        f"base-rate analysis ({result.neighbor_count} neighbors, "
        f"mean similarity {result.mean_similarity:.3f}), the calibrated "
        f"probabilities over outcomes [{outcomes}] are:"
    )
    return f"{explanation}\n{json.dumps(payload)}"


def _build_response(result: PipelineResult, model_name: str) -> dict[str, Any]:
    """Assemble the OpenAI-format chat-completion response."""
    content = _format_message_content(result)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    prompt_tokens = len(result.parsed.question.split())
    completion_tokens = len(content.split())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        # Non-standard but harmless: standard OpenAI parsers ignore unknown
        # top-level keys. Useful for our Streamlit demo.
        "prophet_debug": {
            "request_id": result.request_id,
            "question": result.parsed.question,
            "outcomes": list(result.parsed.outcomes),
            "market_prices": result.parsed.market_prices,
            "resolve_by": result.parsed.resolve_by,
            "p_ensemble": result.p_ensemble,
            "p_ensemble_var": result.p_ensemble_var,
            "base_rate": result.base_rate,
            "neighbor_count": result.neighbor_count,
            "mean_similarity": result.mean_similarity,
            "p_meta": result.p_meta,
            "alpha": result.alpha,
            "p_final": result.p_final,
            "elapsed_s": result.elapsed_s,
            "ensemble_per_model": result.ensemble_raw,
        },
    }


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "prophet-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "prophethacks-2026",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required and non-empty")
    content = _extract_user_content(req.messages)
    if not content:
        raise HTTPException(status_code=400, detail="no user content found in messages")

    # Allow tests / fixtures to swap in fake collaborators via app.state.
    wafer_client = getattr(app.state, "wafer_client", None)
    base_rate_fn = getattr(app.state, "base_rate_fn", None)
    snowflake_logger = getattr(app.state, "snowflake_logger", None)

    try:
        result = await run_pipeline(
            content,
            wafer_client=wafer_client,
            base_rate_fn=base_rate_fn,
            snowflake_logger=snowflake_logger,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"pipeline error: {e}") from e

    return _build_response(result, model_name=req.model)
