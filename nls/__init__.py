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
from .physical import DeepWater, peregrine_focus_series, peregrine_for_height
from .kinematics import (LinearWave, NLSWave, SpectralWaveField, frequency,
                         wave_height, wavenumber)
from .newwave import NewWave, jonswap
from .stream_function import StreamFunctionWave, breaking_height
from .variable_depth import Transect, TransectResult, VariableDepthNLS, irregular_over_transect
from .hos_ocean import HOSCase, SeaState

__all__ = [
    "NLSSolver", "Result",
    "akhmediev", "akhmediev_a_for_period", "background", "breather",
    "breather_params", "kuznetsov_ma", "modulated_background", "peregrine",
    "random_wave_field", "soliton",
    "ROGUE_THRESHOLD", "amplification", "find_rogue_events", "invariant_drift", "invariants",
    "DeepWater", "peregrine_focus_series", "peregrine_for_height",
    "LinearWave", "NLSWave", "SpectralWaveField", "frequency", "wave_height",
    "wavenumber", "NewWave", "jonswap", "StreamFunctionWave", "breaking_height",
    "Transect", "TransectResult", "VariableDepthNLS", "irregular_over_transect",
    "HOSCase", "SeaState",
]
