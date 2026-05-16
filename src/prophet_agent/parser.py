"""Parser for OpenAI-format chat prompts into forecasting questions.

The Prophet Arena harness POSTs OpenAI-compatible chat requests where the user
message contains the forecasting question. The exact format isn't standardized,
so we apply best-effort heuristics:

  * `Question:` / `Forecast:` / leading `Will ...` line   → question text
  * numbered / bulleted list                              → outcomes
  * `Market price:` / `Kalshi price:` / `Market prices:`  → market prices vector
  * `Resolve by:` / `Resolution date:`                    → resolve_by string

If the message doesn't match any of these heuristics, the whole content becomes
the question text and we default to binary `["Yes", "No"]` outcomes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForecastPromptParsed:
    """Structured view of a forecasting prompt."""

    question: str
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    market_prices: list[float] | None = None
    resolve_by: str | None = None
    market_id: str | None = None


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# A line that begins a question section. Captures everything to end-of-line.
_QUESTION_RE = re.compile(
    r"^\s*(?:question|forecast|prompt)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# A bare leading line beginning with "Will ..." — used when no explicit
# Question: marker is present.
_WILL_LINE_RE = re.compile(r"^\s*(Will\s+.+?\??)\s*$", re.IGNORECASE | re.MULTILINE)

# Numbered ("1. Yes") or bulleted ("- Yes" / "* Yes") outcome lines.
_OUTCOME_LINE_RE = re.compile(
    r"^\s*(?:(?:\d+|[a-zA-Z])[.)]|[-*•])\s+(.+?)\s*$",
    re.MULTILINE,
)

# A line like "Outcomes:" / "Options:" / "Choices:" that introduces a list.
_OUTCOMES_HEADER_RE = re.compile(
    r"^\s*(?:outcomes|options|choices|answers)\s*[:\-]\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)

# Single market price, binary case: "Market price: 0.42" or "Kalshi price: 0.42".
_BINARY_PRICE_RE = re.compile(
    r"^\s*(?:market(?:\s+price)?|kalshi(?:\s+price)?|q[_\-\s]?kalshi)\s*[:\-]\s*"
    r"([0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Multi-outcome market prices: "Market prices: 0.3, 0.5, 0.2" or "[0.3, 0.5, 0.2]".
_MULTI_PRICE_RE = re.compile(
    r"^\s*(?:market\s+prices|kalshi\s+prices|q[_\-\s]?kalshi|prices)\s*[:\-]\s*"
    r"\[?\s*([0-9.\s,]+?)\s*\]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_RESOLVE_BY_RE = re.compile(
    r"^\s*(?:resolve(?:s|d)?(?:\s+by)?|resolution(?:\s+date)?|resolves?\s+on)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_MARKET_ID_RE = re.compile(
    r"^\s*(?:market[_\-\s]?id|ticker|market[_\-\s]?ticker)\s*[:\-]\s*([A-Za-z0-9_\-./]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_outcomes(text: str) -> list[str]:
    """Return outcomes found in `text` by list-marker heuristic.

    We accept the first contiguous block of 2+ list-marker lines as the outcome set.
    """
    matches = list(_OUTCOME_LINE_RE.finditer(text))
    if len(matches) < 2:
        return []
    # Find the largest run of matches whose source lines are contiguous (no
    # blank line in between). For simplicity we take the first such run of
    # length >= 2.
    outcomes: list[str] = []
    last_end_line = -2
    for m in matches:
        start_line = text.count("\n", 0, m.start())
        if outcomes and start_line - last_end_line > 2:
            # Non-contiguous — stop, keep what we have.
            break
        outcomes.append(m.group(1).strip())
        last_end_line = start_line
    if len(outcomes) < 2:
        return []
    return outcomes


def _extract_market_prices(text: str, n_outcomes: int) -> list[float] | None:
    """Return a market-price vector of length `n_outcomes`, or None."""
    multi = _MULTI_PRICE_RE.search(text)
    if multi:
        raw = multi.group(1)
        nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", raw)]
        if len(nums) == n_outcomes and all(0.0 <= n <= 1.0 for n in nums):
            return nums

    binary = _BINARY_PRICE_RE.search(text)
    if binary and n_outcomes == 2:
        p = float(binary.group(1))
        if 0.0 <= p <= 1.0:
            return [p, 1.0 - p]
    return None


def _extract_question(text: str) -> str | None:
    """Best-effort question extraction."""
    m = _QUESTION_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _WILL_LINE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def parse_forecast_prompt(content: str) -> ForecastPromptParsed:
    """Parse a Prophet-Arena-style user message into structured fields.

    Falls back to (content, ["Yes","No"], None, None) if nothing parses cleanly.
    """
    if not content or not content.strip():
        return ForecastPromptParsed(question="")

    question = _extract_question(content)

    outcomes = _extract_outcomes(content)
    if not outcomes:
        outcomes = ["Yes", "No"]

    market_prices = _extract_market_prices(content, len(outcomes))

    resolve_by_m = _RESOLVE_BY_RE.search(content)
    resolve_by = resolve_by_m.group(1).strip() if resolve_by_m else None

    market_id_m = _MARKET_ID_RE.search(content)
    market_id = market_id_m.group(1).strip() if market_id_m else None

    if question is None:
        # No recognizable question header. Fall back to the whole content
        # (trimmed) as the question.
        question = content.strip()

    return ForecastPromptParsed(
        question=question,
        outcomes=outcomes,
        market_prices=market_prices,
        resolve_by=resolve_by,
        market_id=market_id,
    )
