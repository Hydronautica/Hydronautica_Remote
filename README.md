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
| `nls.stream_function` | `StreamFunctionWave`: fully nonlinear regular wave by Fenton's Fourier method (analytic Jacobian, automatic mode count, converges to ~98% of breaking); `breaking_height(T, d)` |
| `nls.newwave` | `NewWave`: Tromans focused wave group from a JONSWAP (or custom) spectrum, linear or second order (`order=2`) |
| `nls.second_order` | `SecondOrderWaveField`, `pair_kernels`: Sharma–Dean second-order bound waves (sum and difference) for any unidirectional linear field in finite depth |
| `nls.dysthe` | `DystheSolver` (Dysthe 1979 modified NLS, optional exact dispersion, RK4IP), `DystheWave`, `dysthe_for_height` |
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

## Approaching breaking

`breaking_height(T, d)` gives the limiting height of a steady wave of period T
in depth d. It uses Fenton's fit to Williams' highest waves, with the
wavelength's growth with height taken from the stream function solution. In
deep water this is 0.0269 gT², within 0.4% of the exact value.

`examples/breaking_sweep.py` sweeps H from 0.1 to 0.97 H_b (T = 10 s,
d = 100 m, H_b = 26.4 m) and compares the three models at each height:

- **Stream function**: crest/H rises from 0.51 to 0.66, and u_crest/c from
  0.06 to 0.80. Normalised accelerations and vertical velocities *fall* as the
  wave steepens, because the steep wave has a longer wavelength and flatter
  troughs.
- **NewWave**: being linear, its shape doesn't change (crest/H = 0.577 at every
  height). With Wheeler stretching its crest velocity drops below the stream
  function's above about 0.7 H_b. With linear extrapolation u_crest exceeds the
  phase speed above about 0.75 H_b, i.e. it predicts kinematic breaking.
- **NLS Peregrine**: the crest share rises to 0.70, but not through Stokes-type
  nonlinearity. The envelope narrows to about one wavelength, so the adjacent
  troughs get shallower. Its accelerations grow fastest (+50% over the stream
  function at 0.97 H_b). The background steepness reaches k₀a₀ = 0.24 and
  about 0.2% of the energy sits at k < 0, both well outside where NLS is
  valid.
- **Stretching**: at large H, Wheeler versus linear extrapolation changes the
  crest kinematics of NewWave and NLS by a factor of about 2, more than the
  difference between wave theories.

## Higher-order models: second order and Dysthe

The models above are first order (linear) at two levels: in the *evolution*
(the cubic NLS is the leading-order envelope equation) and in the
*reconstruction* of the surface and kinematics from the envelope or
spectrum. The package can go one order higher in both.

**Second-order bound waves** (`nls.second_order`). A steep wave is not a sum
of independent sinusoids: each pair of components forces bound waves at the
sum and difference frequencies. These make crests sharper and higher, troughs
flatter, and produce a set-down and return flow under a wave group.
`SecondOrderWaveField` adds the Sharma & Dean (1981) solution to any linear
field, so `NewWave(..., order=2)` is second-order NewWave. The kernels are
tested against Stokes second order in finite depth and Tayfun's deep-water
formulas, and the free-surface conditions are checked to be satisfied to
O(a³). For NLS and Dysthe envelopes, `order=2` uses the same deep-water
kernels in envelope form: η₂ = Re[(k₀A²/2 − i/2·A A_x) e^{2iθ}], set-down
η̄ = −¼ F⁻¹[|κ| F|A|²], plus Dysthe's mean-flow potential.

Second-order kinematics use the consistent **Taylor (linear) extrapolation**
above z = 0 by default (`stretching="linear"`). Wheeler stretching is not
consistent with second-order theory.

**The Dysthe equation** (`nls.dysthe`). Dysthe (1979) carried the
envelope expansion one order further than the NLS, to O(ε⁴) in steepness. In
units t′ = ω₀t, x′ = k₀x, B = k₀A, and moving with the group velocity:

```
B_t + i/8 B_ξξ − 1/16 B_ξξξ + i/2 |B|²B + 3/2 |B|² B_ξ + 1/4 B² B*_ξ + i B φ̄_ξ = 0
φ̄_ξ = −½ F⁻¹[|K| F(|B|²)]          (wave-induced mean flow, φ̄_z = ½ ∂_ξ|B|²)
```

The new terms are:

- **third-order dispersion** (−1/16 B_ξξξ): makes the group asymmetric;
- **slope-dependent nonlinearity** (3/2 and 1/4 terms): gives the Stokes
  correction to the group velocity, which is why a steep group runs ahead of
  linear theory;
- **mean-flow coupling**: the return current under a group, which weakens
  modulational instability.

In practice a Dysthe breather focuses *later* and usually *lower* than the
NLS one, with a steep front and a deep trough behind it. `dispersion="exact"`
replaces the polynomial dispersion with the exact deep-water relation
(Trulsen et al. 2000, the "broader bandwidth MNLS"). This matters for steep
focusing, where the envelope is only about a wavelength long.

The solver is checked against several known results:
- the exact uniform Stokes wave;
- the exact Stokes frequency of a detuned wave, to O(ε²K²), which pins the
  3/2 and 1/4 coefficients;
- reduction to NLS when the higher-order terms are switched off;
- 4th-order convergence and conservation of ∫|B|²;
- the mean flow reducing modulational instability growth.

`examples/dysthe_vs_nls.py` compares a Peregrine initial condition (k₀a₀ = 0.1)
under each model:

| model | peak \|A\|/a₀ | time of peak |
|---|---|---|
| NLS | 3.00 | 1.00 t_NLS |
| Dysthe | 2.73 | 1.18 t_NLS |
| Dysthe, exact dispersion | 3.03 | 1.24 t_NLS |
| Dysthe, no mean flow | 2.59 | 0.90 t_NLS |

Both Dysthe variants radiate short envelope ripples ahead of the focus. These
are converged in time step and resolution, so they belong to the model: a sign
that the event is at the edge of narrow-band validity.

### Linear vs second order vs Dysthe towards breaking

`examples/higher_order_sweep.py` calibrates seven models to the same focal
wave height H, for T = 10 s and d = 100 m (H_b = 26.4 m). Dysthe uses the
largest wave near the focus, because its focus becomes multi-peaked.
Selected results:

| H/H_b | model | crest/H | u_crest (m/s) | u_crest/c | max a_x (m/s²) |
|---|---|---|---|---|---|
| 0.4 | Stream function | 0.55 | 4.0 | 0.25 | 2.0 |
| | NewWave linear / 2nd | 0.58 / 0.63 | 4.4 / 5.8 | 0.28 / 0.37 | 2.5 / 3.3 |
| | NLS linear / 2nd | 0.54 / 0.58 | 3.5 / 4.2 | 0.22 / 0.27 | 2.2 / 2.2 |
| | Dysthe / exact-dispersion (2nd) | 0.56 / 0.56 | 4.5 / 4.4 | 0.29 / 0.28 | 2.5 / 2.5 |
| 0.8 | Stream function | 0.62 | 9.8 | 0.55 | 3.7 |
| | NewWave linear / 2nd | 0.58 / 0.67 | 9.1 / 15.1 | 0.58 / 0.97 | 4.2 / 9.2 |
| | NLS linear / 2nd | 0.65 / 0.73 | 8.5 / 13.2 | 0.54 / 0.84 | 5.0 / 6.2 |
| | Dysthe / exact-dispersion (2nd) | 0.63 / 0.61 | 13.1 / 14.5 | 0.84 / 0.93 | 8.0 / 10.6 |

What the sweep shows:

- **Up to about 0.4 H_b the models agree.** Second-order NLS and Dysthe are
  within about 15% of the stream function, and the bound waves give the
  expected sharper crests.
- **Above about 0.5 H_b the extrapolation method dominates.** Taylor
  extrapolation into the crest makes the second-order models' crest
  velocities and accelerations grow much faster than the stream function's.
  This is strongest for broadband NewWave, whose short waves are amplified.
  It is a property of second-order theory near breaking, not of the
  bound-wave terms.
- **Dysthe's crest shape tracks the stream function closely** (crest/H within
  0.02 up to 0.95 H_b), but its envelope needs k₀a₀ ≈ 0.3 at 0.9 H_b, outside
  its range.
- **Every focused-wave model predicts breaking (u > c) by 0.9–0.95 H_b.** For
  structures, that regime needs a slamming load model rather than better
  kinematics.

## Examples

Run from the repo root; figures go to `figures/`.

```bash
python examples/peregrine.py        # Peregrine vs exact, error ~2e-4 at the peak
python examples/akhmediev_mi.py     # MI from a seeded background -> AB + FPUT recurrence
python examples/kuznetsov_ma.py     # 3 KM periods, error ~2e-5
python examples/random_sea.py       # random sea, rogue-event detection
python examples/ocean_peregrine.py  # 10 s swell, 1.5 m: 4.5 m crest, probe time series
python examples/kinematics_comparison.py  # NLS vs stream function vs NewWave kinematics (~35 s)
python examples/breaking_sweep.py         # the same, swept up to 0.97 x breaking height (~2 min)
python examples/dysthe_vs_nls.py          # Peregrine under NLS vs Dysthe (~10 s)
python examples/higher_order_sweep.py     # linear vs 2nd order vs Dysthe, swept towards breaking (10-20 min, 4 processes, cached)
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
