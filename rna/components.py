"""Damage and checks for each RNA component from one load record.

Every component returns a dict with the same keys so a report or a web page can list them:
  key, name, method, standard, ageing (design-life seconds used per second, or None),
  check (a static utilisation where one applies, or None), metrics [(label, text)],
  note, signal (label, unit, series), cum (design-life seconds used up to each sample).

Ageing 1.0 means the part is using its life at exactly the rate that would end it after
20 years. Sizes, ratings and S-N curves below are representative for a 14 MW class
geared turbine, not data for a real machine.
"""
import numpy as np

from .fatigue import DESIGN_LIFE, SN, place, rainflow
from .records import Turbine, shaft_supports


def _result(key, name, method, standard, dmg_t, rec, metrics, note, signal, check=None, ageing=True):
    dur = rec["t"][-1] - rec["t"][0] + rec["dt"]
    cum = np.cumsum(dmg_t) * DESIGN_LIFE if ageing else None
    return dict(key=key, name=name, method=method, standard=standard,
                ageing=float(cum[-1] / dur) if ageing else None, check=check,
                metrics=metrics, note=note, signal=signal, cum=cum)


def _revs(rec):
    return np.abs(rec["rpm"]) / 60 * rec["dt"]


# ---------------------------------------------------------------------------------------------
def main_bearing(rec, tb=None, C=48e6, C0=95e6, e=0.30, Y1=2.3, X2=0.67, Y2=3.4, Y0=2.2):
    """Spherical roller main bearing: ISO 281 equivalent load and L10 life, step by step."""
    tb = tb or Turbine()
    mb, _, fa = shaft_supports(rec, tb)
    Fr = np.hypot(mb[:, 0], mb[:, 1]); Fa = np.abs(fa)
    P = np.where(Fa <= e * Fr, Fr + Y1 * Fa, X2 * Fr + Y2 * Fa)
    L10 = (C / P) ** (10 / 3) * 1e6                       # revolutions
    dmg = _revs(rec) / L10
    s0 = C0 / np.max(Fr + Y0 * Fa)
    moving = rec["rpm"] > 1.0
    low = np.mean(P[moving] < 0.02 * C) if moving.any() else 0.0
    life_y = None if not moving.any() else 1e6 * np.mean((C / P[moving]) ** (10 / 3)) / (np.mean(rec["rpm"][moving]) * 60 * 8766)
    m = [("Radial load (mean)", f"{Fr.mean() / 1e6:.1f} MN"), ("Axial load (mean)", f"{Fa.mean() / 1e6:.1f} MN"),
         ("Equivalent load P/C", f"{P.mean() / C:.2f}"),
         ("L10 if this lasted", f"{life_y:.0f} years" if life_y else "not turning"),
         ("Static safety s0", f"{s0:.1f}"), ("Time under 2 % of C (slip risk)", f"{100 * low:.0f} %")]
    return _result("main_bearing", "Main bearing", "Shaft statics → equivalent load → L10, every time step",
                   "ISO 281, ISO/TS 16281", dmg, rec, m,
                   "Takes the rotor weight lever and all the thrust. Life is used per revolution, so a parked rotor ages it very little; slipping under light load is the field failure to watch.",
                   ("Equivalent load P", "MN", P / 1e6), check=1 / s0)


def main_shaft(rec, tb=None, d_o=1.45, d_i=0.5, Kt=1.5, Kt_t=1.3, Rm=750e6, s_D=160e6, n_D=1e6, m1=5, m2=9):
    """Rotating bending at the main-bearing section with mean torsion (Goodman), FKM-style curve."""
    tb = tb or Turbine()
    a = tb.x_main_bearing - tb.hub[0]
    F, M = rec["hub_F"], rec["hub_M"]
    Mb = np.hypot(M[:, 1] + a * F[:, 2], M[:, 2] - a * F[:, 1])
    Z = np.pi * (d_o ** 4 - d_i ** 4) / (32 * d_o)
    s_a = Kt * Mb / Z                                    # each revolution is one fully reversed cycle
    tau = Kt_t * np.abs(M[:, 0]) / (2 * Z)
    s_eq = s_a / np.maximum(0.05, 1 - np.sqrt(3) * tau / Rm)
    sn = SN(s_D, n_D, m1, m2)
    dmg = _revs(rec) / sn.cycles(s_eq)
    m = [("Bending moment at bearing (max)", f"{Mb.max() / 1e6:.1f} MN·m"), ("Bending stress amplitude (max)", f"{s_a.max() / 1e6:.0f} MPa"),
         ("Torsional stress (mean)", f"{tau.mean() / 1e6:.0f} MPa"), ("Equivalent amplitude / endurance", f"{s_eq.max() / s_D:.2f}")]
    return _result("main_shaft", "Main shaft", "σ = Kt·M/Z per revolution, Goodman for torque",
                   "DIN 743 / FKM", dmg, rec, m,
                   "The bending moment is steady in the fixed frame, so the rotating shaft sees it reversed once per revolution. Sized to stay under its endurance limit, so the ageing is tiny unless a notch or crack raises the stress.",
                   ("Equivalent stress amplitude", "MPa", s_eq / 1e6))


# ---------------------------------------------------------------------------------------------
def design_climate(tb=None, A=11.3, k=2.0, avail=0.97, cut_in=3.0, rated=11.0, cut_out=25.0, turb=0.05):
    """Wind speed bins, probability, rotor speed and torque for the design load spectrum."""
    tb = tb or Turbine()
    U = np.arange(cut_in, cut_out + 0.01, 0.5)
    cdf = lambda u: 1 - np.exp(-(u / A) ** k)
    p = cdf(U + 0.25) - cdf(U - 0.25)
    rpm = np.clip(tb.rated_rpm * U / 10.0, 4.5, tb.rated_rpm)
    P = np.clip((U ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3), 0, 1) * tb.rated_torque * tb.rated_rpm
    T = P / rpm
    return U, p * avail, rpm, T, turb


def _design_rate(m, tb):
    """Mean of rpm·T^m over the design life (Gauss-Hermite over turbulent torque)."""
    U, p, rpm, T, turb = design_climate(tb)
    x, w = np.polynomial.hermite_e.hermegauss(9)
    w = w / w.sum()
    Tt = T[:, None] * (1 + turb * x[None, :])
    return np.sum(p * rpm * np.sum(w * np.abs(Tt) ** m, 1))


def _torque_ageing(rec, tb, m):
    T = np.abs(rec["torque_aero"])
    inst = np.abs(rec["rpm"]) * T ** m / _design_rate(m, tb)     # ageing at each instant
    dur = rec["t"][-1] - rec["t"][0] + rec["dt"]
    return inst, inst * rec["dt"] / DESIGN_LIFE, dur


def gearbox_gears(rec, tb=None, m_bend=8.7, m_pit=6.6):
    """Tooth-root bending and pitting relative to the design spectrum (ISO 6336 slopes)."""
    tb = tb or Turbine()
    ib, db, _ = _torque_ageing(rec, tb, m_bend)
    ip, dp, _ = _torque_ageing(rec, tb, m_pit)
    gov = db if db.sum() >= dp.sum() else dp
    m = [("Torque (mean)", f"{np.mean(np.abs(rec['torque_aero'])) / 1e6:.1f} MN·m"),
         ("Bending ageing (m = 8.7)", f"{ib.mean():.2f}"), ("Pitting ageing (m = 6.6 on torque)", f"{ip.mean():.2f}"),
         ("Rotor speed (mean)", f"{np.mean(rec['rpm']):.1f} rpm")]
    return _result("gears", "Gearbox gears", "Torque and speed vs the design load spectrum",
                   "ISO 6336, IEC 61400-4", gov, rec, m,
                   "Only torque and revolutions matter, so storms barely touch the gears while hours near rated torque use them fastest. Damage is measured against the spectrum the gearbox was designed for, so no gear geometry is needed.",
                   ("Rotor torque", "MN·m", np.abs(rec["torque_aero"]) / 1e6))


def gearbox_bearings(rec, tb=None, m=10 / 3):
    """Planet, intermediate and high-speed bearings: loads scale with torque, cycles with speed."""
    tb = tb or Turbine()
    inst, dmg, _ = _torque_ageing(rec, tb, m)
    mm = [("Torque (mean)", f"{np.mean(np.abs(rec['torque_aero'])) / 1e6:.1f} MN·m"),
          ("Damage-equivalent torque", f"{(np.mean(np.abs(rec['rpm']) * np.abs(rec['torque_aero']) ** m) / max(1e-9, np.mean(np.abs(rec['rpm'])))) ** (1 / m) / 1e6:.1f} MN·m"),
          ("Ageing", f"{inst.mean():.2f}")]
    return _result("gear_bearings", "Gearbox bearings", "L10 with load ∝ torque, relative to the design spectrum",
                   "ISO 281", dmg, rec, mm,
                   "Planet and high-speed bearings fail most often in the field. Their loads come from the gear forces, which follow torque, so the same spectrum method applies with the roller exponent 10/3.",
                   ("Rotor speed", "rpm", rec["rpm"]))


# ---------------------------------------------------------------------------------------------
def _slewing_P(Fr, Fa, M, D):
    """Simplified equivalent axial load for a four-point contact slewing bearing (DG03 style)."""
    return 0.75 * Fr + np.abs(Fa) + 2.0 * M / D


def pitch_bearings(rec, tb=None, D=4.6, Ca=48e6, C0a=110e6, Z=110):
    """Rolling life from pitch travel, static margin, and count of small oscillations (wear risk)."""
    b = rec["blade"]
    theta_crit = 720.0 / Z
    worst, dmg_w, P_w, info = -1, None, None, None
    for k in range(3):
        Fr = np.hypot(b[k, 0], b[k, 1]); Fa = b[k, 2]; M = np.hypot(b[k, 3], b[k, 4])
        P = _slewing_P(Fr, Fa, M, D)
        revs = np.abs(np.diff(rec["pitch"][k], prepend=rec["pitch"][k][0])) / 360
        dmg = revs / ((Ca / P) ** 3 * 1e6)
        rg, _, ct, _ = rainflow(rec["pitch"][k])
        small = ct[(rg > 0.05) & (rg < theta_crit)].sum()
        if dmg.sum() > (dmg_w.sum() if dmg_w is not None else -1):
            worst, dmg_w, P_w, info = k, dmg, P, (M, small, np.abs(np.diff(rec["pitch"][k])).sum())
    M, small, travel = info
    dur = rec["t"][-1] - rec["t"][0] + rec["dt"]
    s0 = C0a / P_w.max()
    m = [("Blade-root bending (mean)", f"{M.mean() / 1e6:.0f} MN·m"), ("Equivalent load P/Ca", f"{P_w.mean() / Ca:.2f}"),
         ("Pitch travel", f"{travel / dur * 3600:.0f} °/h"),
         (f"Oscillations under {theta_crit:.1f}° (wear risk)", f"{small / dur * 3600:.0f} per hour"),
         ("Static safety s0", f"{s0:.1f}")]
    return _result("pitch_bearings", "Pitch bearings", "Equivalent load from blade-root loads; rolling life per degree pitched",
                   "NREL DG03, ISO 281", dmg_w, rec, m,
                   f"Rolling fatigue is small because the bearing only turns a few degrees. The field failures are wear and false brinelling from small back-and-forth motions that never move a ball past its neighbour (under {theta_crit:.1f}° here), so that count matters more than the ageing number. Worst blade: {'ABC'[worst]}.",
                   ("Equivalent load P, blade " + "ABC"[worst], "MN", P_w / 1e6), check=1 / s0)


def blade_root_bolts(rec, tb=None, n=144, D_bc=4.25, d=42, As=1121e-6, F_V=738e3, phi=0.10, s_c=50e6):
    """Bolt force from the joint diagram (gap opening included) at 24 positions per blade, EC3 S-N."""
    ks = min(1.0, (30.0 / d) ** 0.25)
    sn_hi = SN(ks * s_c, 2e6, 3)                                   # above the knee
    s_d = ks * s_c * (2 / 5) ** (1 / 3)
    b = rec["blade"]
    best = None
    for k in range(3):
        for beta in np.radians(np.arange(0, 360, 15)):
            Zb = 4 * (b[k, 3] * np.cos(beta) + b[k, 4] * np.sin(beta)) / (n * D_bc) + b[k, 2] / n
            Fb = np.maximum(F_V + phi * Zb, Zb)
            sig = Fb / As
            rg, _, ct, ends = rainflow(sig)
            Nc = np.where(rg >= s_d, sn_hi.cycles(rg), 5e6 * (s_d / np.maximum(rg, 1e-9)) ** 5)
            dc = ct / Nc
            if best is None or dc.sum() > best[0]:
                best = (dc.sum(), k, beta, sig, rg, ct, ends, dc, (Zb > F_V / (1 - phi)).mean())
    _, k, beta, sig, rg, ct, ends, dc, gap = best
    dmg = place(len(sig), ends, dc)
    m = [("Bolts", f"{n} × M{d}, preload {F_V / 1e3:.0f} kN"), ("Mean bolt stress", f"{sig.mean() / 1e6:.0f} MPa"),
         ("Largest stress range", f"{rg.max() / 1e6:.1f} MPa"), ("Cycles in record", f"{ct.sum():.0f}"),
         ("Joint gap open", f"{100 * gap:.1f} % of the time")]
    r = _result("bolts", "Blade-root bolts", "Joint diagram per bolt position → rainflow → S-N",
                   "VDI 2230, Schmidt–Neuper, EC3 detail 50", dmg, rec, m,
                   f"The blade's own weight swings the edgewise moment once per revolution, so every bolt sees about one stress cycle per turn. Preload keeps only ~{100 * phi:.0f} % of that in the bolt; if preload is lost, the bolt range jumps about tenfold. Worst bolt: blade {'ABC'[k]}, {np.degrees(beta):.0f}° round the circle.",
                   ("Bolt stress, worst position", "MPa", sig / 1e6))
    r["beta"] = float(beta)
    return r


def yaw_bearing(rec, tb=None, D=4.4, Ca=60e6, C0a=160e6):
    """Tower-top loads on the yaw bearing: static margin; rolling life only while yawing."""
    F, M = rec["tt_F"], rec["tt_M"]
    Fa = -F[:, 1]; Fr = np.hypot(F[:, 0], F[:, 2]); Mt = np.hypot(M[:, 0], M[:, 2])
    P = _slewing_P(Fr, Fa, Mt, D)
    revs = np.abs(np.diff(rec["yaw"], prepend=rec["yaw"][0])) / 360
    dmg = revs / ((Ca / P) ** 3 * 1e6)
    s0 = C0a / P.max()
    m = [("Axial load (weight)", f"{Fa.mean() / 1e6:.1f} MN"), ("Tilting moment (max)", f"{Mt.max() / 1e6:.0f} MN·m"),
         ("Yaw torque (max)", f"{np.abs(M[:, 1]).max() / 1e6:.1f} MN·m"), ("Yawed in record", f"{np.abs(np.diff(rec['yaw'])).sum():.0f}°"),
         ("Static safety s0", f"{s0:.1f}")]
    return _result("yaw_bearing", "Yaw bearing", "Tower-top loads → equivalent load; rolling only while yawing",
                   "NREL DG03", dmg, rec, m,
                   "Carries the whole nacelle and the rotor torque as a tilting moment. It turns so rarely that rolling fatigue is negligible; the static margin in storms and wear at standstill decide its design. The yaw torque sets the brake and drive sizing.",
                   ("Equivalent load P", "MN", P / 1e6), check=1 / s0)


def nacelle_equipment(rec, tb=None, limit=2.0):
    """Horizontal nacelle acceleration against an equipment qualification limit."""
    ah = np.hypot(rec["A"][:, 0], rec["A"][:, 2])
    m = [("Peak horizontal acceleration", f"{ah.max():.2f} m/s²"), ("Equipment limit (assumed)", f"{limit:.1f} m/s²"),
         ("Time above half the limit", f"{100 * np.mean(ah > limit / 2):.1f} %")]
    return _result("equipment", "Cabinets, cooling, converter", "Nacelle acceleration vs qualification limit",
                   "Supplier specifications", np.zeros_like(ah), rec, m,
                   "Not a fatigue calculation: electrical and cooling equipment is qualified to an acceleration level, so the check is the peak against that limit.",
                   ("Horizontal acceleration", "m/s²", ah), check=ah.max() / limit, ageing=False)


def unit_load_hotspot(U, L, sn, dirs=25, key="bedplate", name="Bedplate casting", rec=None):
    """Critical-plane fatigue from unit-load stress fields.

    U: (P, C, 6) unit stress tensors [xx, yy, zz, xy, yz, zx] in Pa per unit load at P candidate
    points; L: (C, n) load multipliers. Normal stress on `dirs` planes per point is rainflow
    counted against `sn` (written in stress range); the worst point and plane govern.
    """
    S = np.einsum("pck,cn->pnk", U, L)                  # P × n × 6
    g = np.pi * (3 - np.sqrt(5))                         # Fibonacci points on a hemisphere
    i = np.arange(dirs); z = 1 - (i + 0.5) / dirs; r = np.sqrt(1 - z * z)
    N = np.stack([r * np.cos(g * i), z, r * np.sin(g * i)], 1)
    w = np.stack([N[:, 0] ** 2, N[:, 1] ** 2, N[:, 2] ** 2, 2 * N[:, 0] * N[:, 1], 2 * N[:, 1] * N[:, 2], 2 * N[:, 2] * N[:, 0]], 1)
    sn_all = np.einsum("pnk,dk->pdn", S, w)              # normal stress on each plane
    best = None
    for p in range(S.shape[0]):
        for d in range(dirs):
            rg, _, ct, ends = rainflow(sn_all[p, d])
            dc = ct / sn.cycles(rg)
            if best is None or dc.sum() > best[0]:
                best = (dc.sum(), p, d, rg, ct, ends, dc)
    _, p, d, rg, ct, ends, dc = best
    series = sn_all[p, d]
    dmg = place(series.size, ends, dc)
    m = [("Candidate points checked", f"{S.shape[0]} × {dirs} planes"), ("Largest stress range", f"{rg.max() / 1e6:.0f} MPa"),
         ("Mean normal stress", f"{series.mean() / 1e6:.0f} MPa"), ("Cycles in record", f"{ct.sum():.0f}")]
    res = _result(key, name, "Unit-load FE superposition → critical-plane rainflow → S-N",
                  "DNV-ST-0361 / GL cast S-N", dmg, rec, m,
                  "The FE fields from the nine unit cases are combined with this record's loads; the worst point and plane govern. Castings like this are normally sized by stiffness and extreme loads, so fatigue ageing is low.",
                  ("Normal stress, critical plane at the hot spot", "MPa", series / 1e6))
    res["point"] = int(p)
    res["normal"] = N[d].tolist()
    return res


def not_modelled(key, name, method, standard, why):
    return dict(key=key, name=name, method=method, standard=standard, ageing=None, check=None,
                metrics=[], note=why, signal=None, cum=None)


def assess(rec, tb=None):
    """Every component that can be worked out from the global loads alone."""
    tb = tb or Turbine()
    out = [f(rec, tb) for f in (main_bearing, main_shaft, gearbox_gears, gearbox_bearings,
                                pitch_bearings, blade_root_bolts, yaw_bearing, nacelle_equipment)]
    out.append(not_modelled("hub", "Hub casting", "Unit-load FE with 18 blade-root cases", "DNV-ST-0361",
                            "Needs its own FE model (the same method as the bedplate, with 18 unit cases at the three blade roots). Not built yet."))
    out.append(not_modelled("generator", "Generator bearings", "Bearing life from rotor weight and magnetic pull", "ISO 281",
                            "Its loads are internal (rotor weight, magnetic pull, speed) rather than from the global model. Better tracked by vibration and temperature monitoring."))
    return out
