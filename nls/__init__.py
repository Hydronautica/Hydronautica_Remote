"""Nonlinear Schrödinger solver for breathers and rogue waves."""

from .solver import NLSSolver, Result
from .solutions import (
    akhmediev,
    akhmediev_a_for_period,
    background,
    breather,
    breather_params,
    kuznetsov_ma,
    modulated_background,
    peregrine,
    random_wave_field,
    soliton,
)
from .diagnostics import ROGUE_THRESHOLD, amplification, find_rogue_events, invariant_drift, invariants
from .physical import DeepWater

__all__ = [
    "NLSSolver", "Result",
    "akhmediev", "akhmediev_a_for_period", "background", "breather",
    "breather_params", "kuznetsov_ma", "modulated_background", "peregrine",
    "random_wave_field", "soliton",
    "ROGUE_THRESHOLD", "amplification", "find_rogue_events", "invariant_drift", "invariants",
    "DeepWater",
]
