# nls — NLS solver for breathers and rogue waves

A pseudo-spectral split-step solver for the 1D focusing nonlinear Schrödinger
equation, with exact breather solutions to check it against and a mapping to
deep-water ocean waves.

```
i ψ_t + ½ ψ_xx + |ψ|² ψ = 0          (general: i ψ_t + α ψ_xx + β |ψ|² ψ = 0)
```

## Install

```bash
pip install -e ".[test]"
pytest
```

## Quick start

```python
import numpy as np
import nls

solver = nls.NLSSolver(L=200, N=4096, order=4)
res = solver.run(lambda x: nls.peregrine(x, -4.0), t0=-4.0, t1=4.0, dt=0.002, n_out=401)

res.max_amplitude().max()      # -> 3.000 (Peregrine peak)
nls.invariant_drift(res)       # mass / momentum / Hamiltonian conservation
nls.find_rogue_events(res)     # local maxima above 2√2 × RMS (H > 2 Hs)
```

## What's included

| Module | Contents |
|---|---|
| `nls.solver` | `NLSSolver`: periodic Fourier grid, exact linear and nonlinear sub-steps, Strang (2nd order) or Yoshida (4th order) splitting |
| `nls.solutions` | Exact: `soliton`, `peregrine`, `akhmediev`, `kuznetsov_ma`, general `breather(a)`. Initial conditions: `modulated_background` (Benjamin–Feir seed), `random_wave_field` (Gaussian spectrum, random phases) |
| `nls.diagnostics` | `invariants`, `invariant_drift`, `amplification`, `find_rogue_events` |
| `nls.physical` | `DeepWater`: scales normalised (X, T, ψ) to metres and seconds for a given carrier period/wavelength and amplitude, and rebuilds the surface elevation η |
| `nls.kinematics` | `SpectralWaveField` (linear kinematics of a sum of modes with Wheeler / linear-extrapolation stretching), `NLSWave` (kinematics from an NLS run), `LinearWave`, `wave_height`, dispersion helpers |
| `nls.stream_function` | `StreamFunctionWave`: fully nonlinear regular wave by Fenton's Fourier method |
| `nls.newwave` | `NewWave`: Tromans focused linear wave group from a JONSWAP (or custom) spectrum |
| `nls.plotting` | Space–time maps, 3D surface, peak-amplitude and snapshot plots |

### Breather family

`breather(x, t, a)` uses the Akhmediev–Ankiewicz–Taki parametrisation, with
Ω = 2√(1−2a), b = √(8a(1−2a)) and peak amplitude 1 + 2√(2a):

- `0 < a < ½`: **Akhmediev breather**, periodic in x with period 2π/Ω, localised in t
- `a = ½`: **Peregrine soliton**, localised in both, peak 3
- `a > ½`: **Kuznetsov–Ma breather**, localised in x, periodic in t with period 2π/|b|

`akhmediev_a_for_period(L)` gives the `a` whose breather fits a box of length L.

### Physical units

For a deep-water carrier with wavenumber k₀, frequency ω₀ = √(g k₀) and
background amplitude a₀ (steepness ε = k₀a₀), the envelope A in
η = Re[A e^{i(k₀x − ω₀t)}] is

```
A = a₀ · conj ψ(X, T),   X = √2 k₀ ε (x − c_g t),   T = ω₀ ε² t / 2
```

```python
dw = nls.DeepWater(a0=1.5, period=10.0)
print(dw.describe())
# carrier: lambda=156.1 m, T=10.00 s, a0=1.50 m, k0*a0=0.060; 1 unit X = 291.1 m, 1 unit T = 873.5 s
x, t, eta = dw.surface(psi, X, T, phase=np.pi)
```

## Wave kinematics

`StreamFunctionWave`, `NewWave`, `NLSWave` and `LinearWave` share one interface:

```python
wave.eta(x, t)
kin = wave.kinematics(x, z, t, stretching="wheeler")   # z up from still water level
kin["u"], kin["w"]       # velocities (m/s)
kin["ut"], kin["wt"]     # local accelerations ∂u/∂t, ∂w/∂t (m/s²)
kin["ax"], kin["az"]     # total accelerations Du/Dt, Dw/Dt
kin["ux"], kin["uz"], kin["wx"], kin["wz"]   # velocity gradients
```

Points above the surface or below the bed are NaN.

- **Stream function**: exact to the chosen number of Fourier modes, so it needs no stretching.
- **NewWave and NLS**: linear (Airy) kinematics per mode, with `"wheeler"`, `"linear"` (extrapolation above z = 0) or `"none"` stretching in the crest.
- **NLS local accelerations**: these use the envelope's actual time derivative from the NLS equation, including the Stokes frequency correction, rather than assuming free linear waves.
- **Narrow-band check**: `NLSWave.negative_wavenumber_fraction()` reports how much of the envelope's energy falls outside the narrow-band assumption.
- **Matching heights**: `peregrine_for_height(H, period=T)` picks the background amplitude a₀ so the Peregrine wave has crest-to-trough height H at the focus. NewWave is scaled the same way.

`examples/kinematics_comparison.py` compares all three for H = 12 m, T = 10 s, d = 100 m:

| | crest | trough | u at crest | u(z=0) | u(z=−20) | max a_x | max w |
|---|---|---|---|---|---|---|---|
| Stream function | 6.74 | −5.26 | 4.65 | 3.61 | 1.67 | 2.25 | 3.57 |
| NewWave (linear, Wheeler) | 6.93 | −5.07 | 5.12 | 3.33 | 1.30 | 2.76 | 4.34 |
| NLS Peregrine (Wheeler) | 6.54 | −5.45 | 4.06 | 3.13 | 1.49 | 2.51 | 3.83 |

## Examples

Run from the repo root; figures go to `figures/`.

```bash
python examples/peregrine.py        # Peregrine vs exact, error ~2e-4 at the peak
python examples/akhmediev_mi.py     # MI from a seeded background -> AB + FPUT recurrence
python examples/kuznetsov_ma.py     # 3 KM periods, error ~2e-5
python examples/random_sea.py       # random sea, rogue-event detection
python examples/ocean_peregrine.py  # 10 s swell, 1.5 m: 4.5 m crest, probe time series
python examples/kinematics_comparison.py  # NLS vs stream function vs NewWave kinematics (~35 s)
```

## Practical notes

- **Domain.** The grid is periodic. Akhmediev breathers fit exactly when L is a
  multiple of 2π/Ω. The Peregrine soliton decays only as 1/x², so give it a wide
  box (L ≈ 200); the background is modulationally unstable, so truncation error
  and round-off grow over long runs.
- **Time step.** Error goes as dt⁴ (order 4) or dt² (order 2), and the constant
  scales with peak |ψ|². dt ≈ 1e-3 to 2e-3 is a good start for peaks of 3–4;
  check `invariant_drift` (Hamiltonian drift ≲ 1e-5 is good).
- **Resolution.** You need several grid points across the narrowest peak, whose
  width is ~1/|ψ|max.
- **Model validity.** The cubic NLS is a weakly nonlinear, narrow-band model
  (steepness ≲ 0.1, Δk/k₀ ≪ 1). In random 1D simulations with a large
  Benjamin–Feir index the field turns into long-lived solitons and can reach
  amplitudes that would break in a real sea. Higher-order models (Dysthe / MNLS)
  are the next step for quantitative ocean work.
- **Kinematics of steep breathers.** The NLS and NewWave kinematics are linear
  (first order) in each mode. At the focus of a steep breather the envelope is
  only a wavelength or two long, so check `negative_wavenumber_fraction()`.
  Bound second-order harmonics are not included in either model.
