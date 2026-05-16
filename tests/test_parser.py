"""Unit tests for the prompt parser — no network."""

from __future__ import annotations

import pytest

from prophet_agent.parser import ForecastPromptParsed, parse_forecast_prompt


def test_parse_minimal_question_only() -> None:
    p = parse_forecast_prompt("Question: Will Bitcoin close above $100,000 on 2026-12-31?")
    assert p.question.startswith("Will Bitcoin close above")
    assert p.outcomes == ["Yes", "No"]
    assert p.market_prices is None
    assert p.resolve_by is None


def test_parse_with_outcomes_list_numbered() -> None:
    text = (
        "Question: Who wins the 2026 NBA Finals?\n\n"
        "Outcomes:\n"
        "1. Boston Celtics\n"
        "2. Denver Nuggets\n"
        "3. Other\n"
    )
    p = parse_forecast_prompt(text)
    assert p.question == "Who wins the 2026 NBA Finals?"
    assert p.outcomes == ["Boston Celtics", "Denver Nuggets", "Other"]
    assert p.market_prices is None


def test_parse_with_outcomes_bulleted() -> None:
    text = (
        "Forecast: Will it rain in Chicago tomorrow?\n"
        "- Yes\n"
        "- No\n"
    )
    p = parse_forecast_prompt(text)
    assert p.outcomes == ["Yes", "No"]
    assert "rain in Chicago" in p.question


def test_parse_binary_market_price() -> None:
    text = (
        "Question: Will the S&P close green today?\n"
        "1. Yes\n"
        "2. No\n"
        "Market price: 0.62\n"
    )
    p = parse_forecast_prompt(text)
    assert p.market_prices == pytest.approx([0.62, 0.38])


def test_parse_multi_outcome_prices() -> None:
    text = (
        "Question: Who wins?\n"
        "1. A\n"
        "2. B\n"
        "3. C\n"
        "Market prices: 0.5, 0.3, 0.2\n"
    )
    p = parse_forecast_prompt(text)
    assert p.market_prices == pytest.approx([0.5, 0.3, 0.2])


def test_parse_resolve_by() -> None:
    text = (
        "Question: Will X happen?\n"
        "Resolve by: 2026-12-31\n"
    )
    p = parse_forecast_prompt(text)
    assert p.resolve_by == "2026-12-31"


def test_parse_will_line_no_header() -> None:
    text = "Will inflation exceed 4% in 2026?"
    p = parse_forecast_prompt(text)
    assert p.question.startswith("Will inflation")
    assert p.outcomes == ["Yes", "No"]


def test_parse_fallback_full_content_as_question() -> None:
    text = "Some arbitrary prompt with no recognizable markers and no list."
    p = parse_forecast_prompt(text)
    assert p.question == text.strip()
    assert p.outcomes == ["Yes", "No"]
    assert p.market_prices is None


def test_parse_market_id() -> None:
    text = (
        "Question: Will it happen?\n"
        "Market ID: KXBTC-26DEC31-T100000\n"
    )
    p = parse_forecast_prompt(text)
    assert p.market_id == "KXBTC-26DEC31-T100000"


def test_parse_empty_string() -> None:
    p = parse_forecast_prompt("")
    assert isinstance(p, ForecastPromptParsed)
    assert p.question == ""


def test_parse_ignores_invalid_price() -> None:
    # Multi-outcome price block whose count doesn't match outcomes → ignored.
    text = (
        "Question: Who wins?\n"
        "1. A\n"
        "2. B\n"
        "Market prices: 0.4, 0.3, 0.3\n"
    )
    p = parse_forecast_prompt(text)
    assert p.outcomes == ["A", "B"]
    assert p.market_prices is None
