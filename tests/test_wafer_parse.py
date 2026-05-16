"""Unit tests for Wafer client parsing — no network."""

from __future__ import annotations

import pytest

from prophet_agent.llm.wafer import _parse_probabilities


def test_parse_clean_json() -> None:
    text = 'After thinking: {"probabilities": [0.3, 0.7]}'
    assert _parse_probabilities(text, 2) == pytest.approx([0.3, 0.7])


def test_parse_normalizes() -> None:
    text = '{"probabilities": [0.2, 0.2]}'
    assert _parse_probabilities(text, 2) == pytest.approx([0.5, 0.5])


def test_parse_takes_last_match() -> None:
    text = '{"probabilities": [0.1, 0.9]} ... revised: {"probabilities": [0.4, 0.6]}'
    assert _parse_probabilities(text, 2) == pytest.approx([0.4, 0.6])


def test_parse_rejects_wrong_arity() -> None:
    text = '{"probabilities": [0.5, 0.5]}'
    with pytest.raises(ValueError):
        _parse_probabilities(text, 3)


def test_parse_rejects_missing() -> None:
    text = "I have no idea what to do."
    with pytest.raises(ValueError):
        _parse_probabilities(text, 2)
