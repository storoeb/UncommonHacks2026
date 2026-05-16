"""Unit tests for the pure-Python helpers in scripts/embed_questions.py."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/embed_questions.py as a module without requiring `scripts/` to
# be a package.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "embed_questions.py"
_spec = importlib.util.spec_from_file_location("embed_questions", _SCRIPT_PATH)
assert _spec and _spec.loader, "could not locate embed_questions.py"
embed_questions = importlib.util.module_from_spec(_spec)
sys.modules["embed_questions"] = embed_questions
_spec.loader.exec_module(embed_questions)  # type: ignore[union-attr]


def test_positive_int_accepts_positive() -> None:
    assert embed_questions.positive_int("1") == 1
    assert embed_questions.positive_int("500") == 500


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        embed_questions.positive_int("0")


def test_positive_int_rejects_negative() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        embed_questions.positive_int("-3")


def test_positive_int_rejects_non_integer() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        embed_questions.positive_int("abc")


def test_parse_args_defaults() -> None:
    ns = embed_questions.parse_args([])
    assert ns.batch_size == 500
    assert ns.max_batches == 50


def test_parse_args_custom() -> None:
    ns = embed_questions.parse_args(["--batch-size", "50", "--max-batches", "1"])
    assert ns.batch_size == 50
    assert ns.max_batches == 1
