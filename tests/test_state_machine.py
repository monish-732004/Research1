"""Unit tests for job state-machine cleaning."""

import pytest
from src.preprocess.state_machine import is_legal


def test_legal_waiting_to_running():
    assert is_legal("Waiting", "Running") is True


def test_legal_running_to_succeeded():
    assert is_legal("Running", "Succeeded") is True


def test_illegal_running_to_waiting():
    assert is_legal("Running", "Waiting") is False


def test_illegal_succeeded_to_running():
    assert is_legal("Succeeded", "Running") is False


def test_unknown_src_state():
    assert is_legal("Unknown", "Running") is False
