"""Scour physics at a monopile under waves and current.

Engineering-level formulas, chosen to answer two questions from a metocean
record: *when does scour start* (onset and live-bed thresholds) and *how bad
does it get* (equilibrium depth, development in time, backfilling).

References
----------
Soulsby (1997) Dynamics of Marine Sands; Soulsby & Whitehouse (1997) threshold
of motion; Soulsby (1995) wave-current bed shear (DATA2 fit);
Sumer, Fredsøe & Christiansen (1992) scour around a pile in waves / current;
Sumer & Fredsøe (2001, 2002) combined waves and current;
Melville & Sutherland (1988) flow-intensity factor for clear-water scour;
Whitehouse (1998) Scour at Marine Structures; DNV-RP-C212 / DNV-ST-0126 practice.

Simplifications (documented in the README): a single sediment size, the
horseshoe-vortex amplification of bed shear taken as 4 for onset, a smooth
blend from the wave-limited Sumer-Fredsøe depth to the current-alone value
for current-dominated flow (Ucw 0.7 to 1), and a linear relaxation of the
scour depth to the instantaneous equilibrium with the Sumer time scale.
"""

from dataclasses import dataclass

import numpy as np

G = 9.81
RHO_W = 1025.0
NU = 1.36e-6          # seawater kinematic viscosity at about 10 degC (m^2/s)
KAPPA = 0.40
AMPLIFICATION = 4.0   # bed shear amplification at the pile (horseshoe vortex, Sumer 2002)
S_D_CURRENT = 1.3     # current-alone live-bed equilibrium scour depth / D


@dataclass
class Sediment:
    d50: float = 0.25e-3        # median grain size (m)
    rho_s: float = 2650.0       # grain density (kg/m^3)

    @property
    def s(self):
        return self.rho_s / RHO_W

    @property
    def d_star(self):
        """Dimensionless grain size D*."""
        return self.d50 * (G * (self.s - 1) / NU**2) ** (1 / 3)

    @property
    def theta_cr(self):
        """Critical Shields parameter (Soulsby & Whitehouse 1997)."""
        ds = self.d_star
        return 0.30 / (1 + 1.2 * ds) + 0.055 * (1 - np.exp(-0.020 * ds))

    @property
    def tau_cr(self):
        return self.theta_cr * G * (self.rho_s - RHO_W) * self.d50

    def shields(self, tau):
        return np.asarray(tau) / (G * (self.rho_s - RHO_W) * self.d50)

    @property
    def z0(self):
        """Grain roughness length, k_s = 2.5 d50, z0 = k_s / 30."""
        return self.d50 / 12.0


# --- hydrodynamics at the bed ------------------------------------------------------

def wavenumber(omega, h):
    """Linear dispersion, solved by Newton iteration (vectorised)."""
    omega = np.asarray(omega, float)
    k = omega**2 / G / np.sqrt(np.tanh(omega**2 * h / G))   # Eckart start
    for _ in range(30):
        th = np.tanh(k * h)
        f = G * k * th - omega**2
        df = G * th + G * k * h * (1 - th**2)
        k = k - f / df
    return k


def _um_per_hs(tp, h, gamma=3.3, n=160):
    """U_m for Hs = 1 m at one peak period (linear theory over a JONSWAP spectrum)."""
    fp = 1 / tp
    f = np.linspace(0.5 * fp, 4.0 * fp, n)
    sig = np.where(f <= fp, 0.07, 0.09)
    S = f**-5 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** np.exp(-((f - fp) ** 2) / (2 * sig**2 * fp**2))
    df = f[1] - f[0]
    S *= (1 / 4) ** 2 / (S.sum() * df)
    w = 2 * np.pi * f
    kh = np.minimum(wavenumber(w, h) * h, 700)
    su = S * (w / np.sinh(kh)) ** 2
    return np.sqrt(2) * np.sqrt(2 * su.sum() * df)


_UM_TABLE = {}


def bed_orbital_velocity(hs, tp, h, gamma=3.3):
    """Significant bed orbital velocity amplitude U_m = sqrt(2) U_rms from a
    JONSWAP spectrum and linear theory. U_m is linear in Hs for a given spectral
    shape, so a per-depth table over Tp makes long records cheap."""
    key = (round(float(h), 3), gamma)
    if key not in _UM_TABLE:
        grid = np.linspace(2.0, 25.0, 231)
        _UM_TABLE[key] = (grid, np.array([_um_per_hs(t, h, gamma) for t in grid]))
    grid, per = _UM_TABLE[key]
    hs, tp = np.broadcast_arrays(np.asarray(hs, float), np.asarray(tp, float))
    return np.where((hs > 0) & (tp > 0), np.maximum(hs, 0) * np.interp(tp, grid, per), 0.0)


def current_shear(u, h, sed):
    """Bed shear from a depth-averaged current with a logarithmic profile."""
    cd = (KAPPA / (np.log(h / sed.z0) - 1)) ** 2
    return RHO_W * cd * np.asarray(u, float) ** 2


def current_at(u, h, z, sed):
    """Velocity at height z above the bed from the depth-averaged value."""
    return np.asarray(u, float) * np.log(np.maximum(z, sed.z0 * 1.01) / sed.z0) / (np.log(h / sed.z0) - 1)


def wave_shear(um, tp, sed):
    """Wave bed shear amplitude with Soulsby's rough-bed friction factor."""
    um = np.asarray(um, float)
    a = um * np.asarray(tp, float) / (2 * np.pi)
    fw = np.where(a > 0, 1.39 * (np.maximum(a, 1e-9) / sed.z0) ** -0.52, 0.0)
    return 0.5 * RHO_W * fw * um**2


def combined_shear(tau_c, tau_w, phi=0.0):
    """Mean and maximum bed shear under waves plus current (Soulsby 1995)."""
    tau_c, tau_w = np.asarray(tau_c, float), np.asarray(tau_w, float)
    tot = np.maximum(tau_c + tau_w, 1e-12)
    tau_m = tau_c * (1 + 1.2 * (tau_w / tot) ** 3.2)
    tau_max = np.sqrt((tau_m + tau_w * np.cos(phi)) ** 2 + (tau_w * np.sin(phi)) ** 2)
    return tau_m, tau_max


# --- pile scour ------------------------------------------------------------------------

@dataclass
class Pile:
    D: float = 10.0             # pile diameter (m)
    h: float = 40.0             # water depth (m)
    sediment: Sediment = None

    def __post_init__(self):
        if self.sediment is None:
            self.sediment = Sediment()


def live_bed_depth(kc, ucw):
    """Equilibrium scour depth / D in live-bed conditions.

    Sumer & Fredsøe (2001) for combined waves and current,
    S/D = 1.3 {1 - exp[-A (KC - B)]}, A = 0.03 + 0.75 Ucw^2.6, B = 6 exp(-4.7 Ucw),
    blended smoothly to the current-alone value 1.3 as the flow becomes current
    dominated (Ucw from 0.5 to 0.9). Large monopiles sit at KC < 4, below the wave
    formula's calibration, where it would otherwise switch scour off abruptly.
    """
    kc, ucw = np.broadcast_arrays(np.asarray(kc, float), np.clip(np.asarray(ucw, float), 0, 1))
    A = 0.03 + 0.75 * ucw**2.6
    B = 6 * np.exp(-4.7 * ucw)
    wave = np.where(kc > B, 1 - np.exp(-A * (kc - B)), 0.0)
    blend = np.clip((ucw - 0.5) / 0.4, 0, 1)
    return S_D_CURRENT * np.maximum(wave, blend * blend * (3 - 2 * blend))


def intensity_factor(theta, theta_cr):
    """Clear-water reduction (Melville & Sutherland), with V/Vc ~ sqrt(theta/theta_cr):
    0 at scour onset at the pile (theta = theta_cr / 4), 1 at live-bed (theta = theta_cr)."""
    r = np.sqrt(np.maximum(np.asarray(theta, float), 0) / theta_cr)
    return np.clip(2 * r - 1, 0, 1)


def depth_factor(h, D):
    """Shallow-water reduction of the equilibrium depth (smooth, ~1 for h/D > 2.5)."""
    return np.tanh(np.asarray(h, float) / D)


def time_scale(theta, kc, ucw, pile, current_only=False):
    """Scour time scale T (s) (Sumer et al. 1992): T = T* D^2 / sqrt(g (s-1) d50^3), with
    T*_current = (1/2000) (h/D) theta^-2.2 and T*_waves = 1e-6 (KC/theta)^3. The wave
    form is only used in the wave-scour regime (KC > 4, interpolated in log space by
    Ucw); below that, as for large monopiles, the current form with the combined
    Shields number applies."""
    sed = pile.sediment
    th = np.maximum(np.asarray(theta, float), 0.005)
    tc = (1 / 2000) * (pile.h / pile.D) * th**-2.2
    if current_only:
        tstar = tc
    else:
        kc = np.asarray(kc, float)
        tw = 1e-6 * (np.maximum(kc, 0.1) / th) ** 3
        w = (1 - np.clip(np.asarray(ucw, float), 0, 1)) * np.clip((kc - 4) / 4, 0, 1)
        tstar = np.exp((1 - w) * np.log(tc) + w * np.log(np.maximum(tw, 1e-9)))
    return tstar * pile.D**2 / np.sqrt(G * (sed.s - 1) * sed.d50**3)


def assess(pile, uc, hs, tp, phi=0.0):
    """Instantaneous scour state for depth-averaged current uc and sea state (hs, tp).

    Returns a dict of arrays: um, kc, ucw, tau_c, tau_w, tau_max, theta (far field),
    onset (scour active at the pile), live (bed mobile in the far field),
    s_eq (equilibrium scour depth, m), T (time scale, s).
    """
    sed = pile.sediment
    uc, hs, tp = np.broadcast_arrays(np.abs(np.asarray(uc, float)), np.asarray(hs, float), np.asarray(tp, float))
    um = bed_orbital_velocity(hs, tp, pile.h)
    ucd = current_at(uc, pile.h, pile.D / 2, sed)          # undisturbed current at D/2 above the bed
    kc = um * tp / pile.D
    ucw = np.where(ucd + um > 0, ucd / np.maximum(ucd + um, 1e-9), 0.0)
    tau_c = current_shear(uc, pile.h, sed)
    tau_w = wave_shear(um, tp, sed)
    _, tau_max = combined_shear(tau_c, tau_w, phi)
    theta = sed.shields(tau_max)
    onset = AMPLIFICATION * theta > sed.theta_cr
    live = theta > sed.theta_cr
    s_eq = pile.D * live_bed_depth(kc, ucw) * intensity_factor(theta, sed.theta_cr) * depth_factor(pile.h, pile.D)
    T = time_scale(theta, kc, ucw, pile)
    T_bf = time_scale(theta, kc, ucw, pile, current_only=True)
    return dict(um=um, ucd=ucd, kc=kc, ucw=ucw, tau_c=tau_c, tau_w=tau_w, tau_max=tau_max,
                theta=theta, onset=onset, live=live, s_eq=s_eq, T=T, T_bf=T_bf)


def simulate(pile, dt_hours, uc, hs, tp, s0=0.0, backfill_factor=3.0):
    """Integrate the scour depth through a metocean record (equal steps of dt_hours).

    Growth: S relaxes to the instantaneous equilibrium with time scale T.
    Backfill: when S exceeds the equilibrium and the far-field bed is mobile,
    S relaxes down with backfill_factor x the current-form time scale (backfilling
    is slower than scouring, Sumer et al. 2013). Otherwise S is frozen.
    Returns (S, state) with S in metres at the end of each step.
    """
    st = assess(pile, uc, hs, tp)
    dt = dt_hours * 3600.0
    S = np.empty(len(st["s_eq"]))
    s = s0
    for i, (seq, T, Tb, live) in enumerate(zip(st["s_eq"], st["T"], st["T_bf"], st["live"])):
        if seq > s:
            s += (seq - s) * (1 - np.exp(-dt / T))
        elif live:
            s += (seq - s) * (1 - np.exp(-dt / (backfill_factor * Tb)))
        S[i] = s
    return S, st


def onset_current(pile, hs=0.0, tp=8.0, level="onset"):
    """Depth-averaged current (m/s) at which scour starts at the pile ("onset") or
    the far-field bed becomes mobile ("live"), for a given sea state."""
    target = pile.sediment.theta_cr / (AMPLIFICATION if level == "onset" else 1.0)
    lo, hi = 0.0, 5.0
    if assess(pile, lo, hs, tp)["theta"] >= target:
        return 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if assess(pile, mid, hs, tp)["theta"] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def armour_utilisation(rock_d50, uc, hs, tp, h, rho_r=2650.0, amplification=2.0):
    """Stability of a rock scour protection: amplified Shields number on the armour
    divided by its critical value (above 1: armour stones can move). Amplification
    about 2 on top of a protection layer (reduced from 4 at a bare pile)."""
    rock = Sediment(d50=rock_d50, rho_s=rho_r)
    um = bed_orbital_velocity(hs, tp, h)
    tau_c = current_shear(uc, h, rock)
    tau_w = wave_shear(um, tp, rock)
    _, tau_max = combined_shear(tau_c, tau_w)
    return amplification * rock.shields(tau_max) / rock.theta_cr
