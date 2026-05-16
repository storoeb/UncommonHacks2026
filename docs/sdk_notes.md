# Prophet Arena agent — HTTP contract notes

Reference for the **OpenAI-compatible** surface our FastAPI agent exposes. Intended audience: judges, integrators, and anyone aligning prompts with `src/prophet_agent/parser.py`.

**Upstream SDK:** [`github.com/ai-prophet/ai-prophet`](https://github.com/ai-prophet/ai-prophet). We **did** pull the public [`docs/build_a_bot.md`](https://github.com/ai-prophet/ai-prophet/blob/main/docs/build_a_bot.md) and repo README via web fetch (May 2026). Those documents describe **`ai-prophet-core`**’s **BenchmarkSession** / trading tick lifecycle—not the JSON body of an OpenAI **`chat.completions`** call to a **custom agent URL**. We have **not** run the Prophet Arena harness end-to-end against this repo nor verified byte-for-byte compatibility with whatever the `prophet trade eval` CLI emits for custom HTTP agents. **Treat this file plus our code as the authoritative contract for our agent** until cross-checked against a released example bot.

---

## 1. Endpoint contract

- **Method / path:** `POST /v1/chat/completions`
- **Shape:** OpenAI **Chat Completions** JSON (permissive parser: unknown top-level fields allowed on the request model in our implementation).

**Example request** (from `tests/fixtures/sample_event.json`):

```json
{
  "model": "prophet-agent",
  "messages": [
    {
      "role": "system",
      "content": "You are a probabilistic forecaster. Reason carefully and return calibrated probabilities."
    },
    {
      "role": "user",
      "content": "Question: Will Bitcoin close above $100,000 USD on December 31, 2026?\n\nOutcomes:\n1. Yes\n2. No\n\nMarket price: 0.55\nResolve by: 2026-12-31\nMarket ID: KXBTC-26DEC31-T100000\n\nRespond with calibrated probabilities."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 512
}
```

The server uses the **last `user` message with non-empty `content`** as the forecasting prompt; if none exists, it falls back to the last message with content. See `src/prophet_agent/server.py` (`_extract_user_content`).

---

## 2. What the user message can look like (parser heuristics)

Implementation: `parse_forecast_prompt()` in `src/prophet_agent/parser.py`. These are **best-effort regexes**, not a formal grammar.

### Question text

- **Explicit headers** — lines matching `Question:`, `Forecast:`, or `Prompt:` (case-insensitive, `:` or `-` after the keyword). Regex: ```38:41:src/prophet_agent/parser.py
_QUESTION_RE = re.compile(
    r"^\s*(?:question|forecast|prompt)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
```

- **Leading “Will …” line** — if no header matched, a line starting with `Will ` (through optional `?`) is treated as the question: ```43:45:src/prophet_agent/parser.py
_WILL_LINE_RE = re.compile(r"^\s*(Will\s+.+?\??)\s*$", re.IGNORECASE | re.MULTILINE)
```

### Outcomes

- Parsed from a **contiguous block** of **two or more** lines that look like numbered or bulleted list items (`1. Yes`, `- No`, `a) Maybe`, etc.). Regex for each line: ```47:50:src/prophet_agent/parser.py
_OUTCOME_LINE_RE = re.compile(
    r"^\s*(?:(?:\d+|[a-zA-Z])[.)]|[-*•])\s+(.+?)\s*$",
    re.MULTILINE,
)
```

- Optional section headers `Outcomes:`, `Options:`, `Choices:`, `Answers:` are recognized but extraction still relies on the list-marker lines: ```53:56:src/prophet_agent/parser.py
_OUTCOMES_HEADER_RE = re.compile(
    r"^\s*(?:outcomes|options|choices|answers)\s*[:\-]\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)
```

- If fewer than two outcomes are found, the parser defaults to **`["Yes", "No"]`**: ```147:149:src/prophet_agent/parser.py
    outcomes = _extract_outcomes(content)
    if not outcomes:
        outcomes = ["Yes", "No"]
```

### Market prices (`q_kalshi`)

| Case | Pattern | Notes |
|------|---------|------|
| **Binary** | `Market price: 0.42`, `Kalshi price: …`, `q_kalshi: …` | Parsed single probability in `[0,1]`; second outcome gets `1 - p`. ```59:64:src/prophet_agent/parser.py |
| **Multi-outcome** | `Market prices: 0.3, 0.5, 0.2` or bracketed list | All values must be in `[0,1]` and **count must match** outcome count. ```66:70:src/prophet_agent/parser.py |

### Resolve-by date

- Lines like `Resolve by: 2026-12-31`, `Resolution date: …`, `Resolves on: …`, etc.: ```73:76:src/prophet_agent/parser.py
_RESOLVE_BY_RE = re.compile(
    r"^\s*(?:resolve(?:s|d)?(?:\s+by)?|resolution(?:\s+date)?|resolves?\s+on)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
```

### Market ID

- `Market ID: …`, `Ticker: …`, `Market ticker: …` — alphanumeric / `-` / `_` / `.` / `/`: ```78:81:src/prophet_agent/parser.py
_MARKET_ID_RE = re.compile(
    r"^\s*(?:market[_\-\s]?id|ticker|market[_\-\s]?ticker)\s*[:\-]\s*([A-Za-z0-9_\-./]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
```

### Fallback behavior

If no question header or `Will` line matches, **the entire message body** becomes the `question` string (trimmed): ```159:162:src/prophet_agent/parser.py
    if question is None:
        # No recognizable question header. Fall back to the whole content
        # (trimmed) as the question.
        question = content.strip()
```

See module docstring at top of `parser.py` for the human-readable summary.

---

## 3. Response shape

The handler returns an OpenAI-style **`chat.completion`** object plus top-level **`prophet_debug`**.

Assistant text is formatted in `_format_message_content()` — short prose, then **`json.dumps({"probabilities": probs})`** on its own line:

```69:81:src/prophet_agent/server.py
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
```

Full assembly (choices, usage, debug block) is in `_build_response()`:

```84:127:src/prophet_agent/server.py
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
```

**Payload inside `content` (after the prose line):**

```json
{"probabilities": [0.523, 0.477]}
```

---

## 4. The `probabilities` JSON convention

The assistant `content` **must** include a JSON object of the form `{"probabilities": [<float>, ...]}` so automated graders can scrape final probabilities.

A **practical** harness regex (assumption—not copied from ai-prophet source here):

```python
r'\{"probabilities":\s*\[[^\]]+\]\}'
```

**Our implementation:** `_format_message_content` appends `json.dumps({"probabilities": probs})` on its **own line** after the explanation, so the above pattern matches the **`content`** string we emit **today**. If you change prose formatting to include additional `[` / `]` brackets before the payload, re-validate any naive regex.

---

## 5. Caveats and known gaps

| Gap | Detail |
|-----|--------|
| **ai-prophet SDK parity** | Public docs emphasize **`BenchmarkSession` + trading intents** (`build_a_bot.md`). We did **not** locate an official “OpenAI chat completions body for custom agents” spec in that fetch. |
| **Heuristic parser** | Headers, lists, and prices are regex-driven; adversarial or free-form prompts may collapse to “whole body = question; Yes/No default outcomes.” |
| **Non-standard fields** | `prophet_debug` is **not** OpenAI-standard; clients must tolerate unknown keys if they mirror OpenAI clients strictly. |
| **Streaming** | Request may include `"stream": false` (fixture); streaming responses are **not** implemented here. |

---

## 6. `curl` against a local agent

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "@tests/fixtures/sample_event.json"
```

Health: `GET http://127.0.0.1:8000/healthz` → `{"status":"ok"}`.
