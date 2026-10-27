"""Conserved quantities and rogue-wave statistics."""

import numpy as np


def invariants(psi, dx, alpha=0.5, beta=1.0):
    """Mass, momentum and Hamiltonian of a periodic field.

    For i psi_t + alpha psi_xx + beta |psi|^2 psi = 0 these are exactly
    conserved; their drift is a cheap accuracy check on a simulation.
    """
    psi = np.asarray(psi)
    N = psi.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_x = np.fft.ifft(1j * k * np.fft.fft(psi, axis=-1), axis=-1)
    mass = np.sum(np.abs(psi) ** 2, axis=-1) * dx
    momentum = np.sum(np.imag(np.conj(psi) * psi_x), axis=-1) * dx
    hamiltonian = np.sum(alpha * np.abs(psi_x) ** 2
                         - 0.5 * beta * np.abs(psi) ** 4, axis=-1) * dx
    return {"mass": mass, "momentum": momentum, "hamiltonian": hamiltonian}


def invariant_drift(result):
    """Max relative drift of each invariant over a :class:`nls.solver.Result`.

    Momentum is often ~0 initially, so its drift is reported relative to the
    mass instead (i.e. as a drift in the mean wavenumber).
    """
    p = result.params
    dx = result.x[1] - result.x[0]
    inv = invariants(result.psi, dx, p.get("alpha", 0.5), p.get("beta", 1.0))
    drift = {}
    for name, v in inv.items():
        scale = inv["mass"][0] if name == "momentum" else max(abs(v[0]), 1e-300)
        drift[name] = float(np.max(np.abs(v - v[0])) / scale)
    return drift


def amplification(result, reference=None):
    """Peak |psi| at each time divided by a reference amplitude.

    The reference defaults to the RMS amplitude of the initial field. A
    Peregrine soliton reaches 3, the usual rogue-wave threshold is ~2.
    """
    if reference is None:
        reference = np.sqrt(np.mean(np.abs(result.psi[0]) ** 2))
    return result.max_amplitude() / reference


ROGUE_THRESHOLD = 2 * np.sqrt(2)


def find_rogue_events(result, threshold=ROGUE_THRESHOLD, reference=None):
    """Locate local maxima of |psi| in (x, t) exceeding threshold * reference.

    The reference defaults to the RMS of |psi| at t=0. For a narrow-banded sea
    the significant wave height is Hs = 4 std(eta) = 2 sqrt(2) RMS|A| and the
    local wave height is ~2|A|, so the oceanographic criterion H > 2 Hs
    becomes |A| > 2 sqrt(2) RMS|A| -- the default threshold.

    Returns a list of dicts with keys t, x, amplitude, amplification, sorted by
    amplitude (largest first).
    """
    amp = result.amplitude
    if reference is None:
        reference = np.sqrt(np.mean(amp[0] ** 2))
    # A point is a peak if it beats its 8 neighbours (periodic in x).
    padded = np.pad(amp, ((1, 1), (0, 0)), mode="constant", constant_values=-np.inf)
    is_peak = np.ones_like(amp, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(padded, dj, axis=1)[1 + di: 1 + di + amp.shape[0]]
            is_peak &= amp >= shifted
    is_peak &= amp > threshold * reference
    it, ix = np.nonzero(is_peak)
    events = [dict(t=float(result.t[i]), x=float(result.x[j]),
                   amplitude=float(amp[i, j]),
                   amplification=float(amp[i, j] / reference))
              for i, j in zip(it, ix)]
    return sorted(events, key=lambda e: -e["amplitude"])
