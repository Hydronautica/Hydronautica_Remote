"""Fatigue history since commissioning for the StormTwin demo (synthetic).

1. Damage-rate table: for every turbine, run the screening chain (NLS sea ->
   Morison -> nonlinear global model -> hot spots -> rainflow / S-N) for a set
   of sea states, giving damage per hour at each hot spot as a function of Hs
   (Tp follows a steepness relation).
2. Sea-state history: a synthetic hourly Hs record at the farm from
   commissioning up to the current storm, with seasonal variation and named
   storms; each turbine sees it scaled by its exposure.
3. Accumulation: damage per hour interpolated from the table and summed,
   then the current storm's detailed result (demo_data.json) is added.

Wave-induced fatigue with an idling rotor only: in production the operational
aerodynamic loads and damping come from SCADA and the aero-elastic model.

Run after build_data.py:  python build_history.py   (~15-20 min, 4 processes)
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_data as bd  # noqa: E402
from structure import Model, Turbine, bolt_damage, miner_damage, rainflow  # noqa: E402

HS_BINS = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
COMMISSIONED = datetime(2019, 3, 1)
STORM_PEAK = datetime(2026, 1, 14, 6)        # the storm replayed on the page
HOT = ("flange_bolts", "door_frame", "mudline_weld")
REF_HS = 9.2                                 # farm-reference Hs of the replayed storm


def tp_of(hs):
    """Peak period from a typical wind-sea steepness relation."""
    return max(5.5, 3.6 * np.sqrt(hs))


def damage_rates(args):
    """Damage per hour at each hot spot for one turbine and one sea state."""
    import warnings
    warnings.filterwarnings("ignore")
    spec, hs = args
    tb = Turbine(depth=spec["depth"], scour=spec["scour"], preload_frac=spec["preload"],
                 name=spec["name"])
    model = Model(tb)
    model.setup_damping()
    s = dict(spec, Hs=hs, seed=spec["seed"] + int(10 * hs))
    dw, solver, res = bd.nls_sea(s, Tp=tp_of(hs))
    t1, A1 = bd.envelope_at(dw, res)
    t1 = t1 - t1[0]
    t = np.arange(0, bd.RECORD + 1e-9, bd.DT)
    csr, csi = CubicSpline(t1, A1.real), CubicSpline(t1, A1.imag)
    A = csr(t) + 1j * csi(t)
    At = csr(t, 1) + 1j * csi(t, 1)
    eta, u, ax, _ = bd.screening_kinematics(tb, model, t, A, At, dw)
    F = bd.morison_nodal(model, eta, u, ax)
    F *= (0.5 - 0.5 * np.cos(np.pi * np.clip(t / 20.0, 0, 1)))[:, None]
    U, _ = model.run(t, F)
    z_mw = bd.model_mw(model, U, tb)
    hsd = bd.hot_spots(tb, model, U, z_mw)
    keep = t >= 30.0                                   # skip the start-up transient
    per_hour = 3600.0 / (bd.RECORD - 30.0)
    # moment statistics at the hot-spot sections, for the browser what-if model
    stats = {}
    for key, M in (("flange", hsd["flange"]["M"]), ("door", hsd["door"]["M"]),
                   ("mudline", hsd["mudline"]["M"])):
        m = M[keep]
        f = m - m.mean()
        nu = np.sum((f[:-1] < 0) & (f[1:] >= 0)) / (bd.RECORD - 30.0)
        stats[key] = dict(mean=float(m.mean()), std=float(f.std()), nu=float(nu))
    r, c = rainflow(hsd["flange"]["Fb"][keep] / tb.bolt_As)
    out = {"flange_bolts": bolt_damage(r, c, tb.bolt_d) * per_hour}
    for key, name, curve in (("door", "door_frame", "D_air"),
                             ("mudline", "mudline_weld", "D_seawater_cp")):
        r, c = rainflow(hsd[key]["s"][keep])
        out[name] = miner_damage(r, c, curve, t=hsd[key]["t"]) * per_hour
    print(f"{spec['name']} Hs={hs:4.1f} Tp={tp_of(hs):4.1f}: "
          + " ".join(f"{k}={v:.2e}" for k, v in out.items()), flush=True)
    props = dict(z_mw=z_mw, f1=float(model.frequencies(n=1)[0]))
    return spec["name"], hs, out, stats, props


# --- synthetic sea-state history ----------------------------------------------------

NAMED_STORMS = [  # (peak time, peak Hs at the farm reference, half-width in hours)
    (datetime(2019, 12, 9, 3), 7.4, 20),
    (datetime(2020, 2, 15, 18), 8.6, 26),
    (datetime(2020, 2, 23, 9), 7.1, 16),
    (datetime(2021, 1, 28, 12), 7.8, 22),
    (datetime(2022, 2, 18, 15), 8.9, 24),
    (datetime(2023, 11, 2, 6), 7.6, 18),
    (datetime(2024, 12, 7, 21), 8.2, 22),
    (datetime(2025, 10, 30, 0), 6.9, 16),
]


def hs_history(seed=7):
    """Hourly Hs at the farm reference point from commissioning to the storm peak
    (the replayed storm itself is added separately from its detailed run)."""
    rng = np.random.default_rng(seed)
    n = int((STORM_PEAK - COMMISSIONED).total_seconds() // 3600) - 12
    times = np.array([COMMISSIONED + timedelta(hours=h) for h in range(n)])
    doy = np.array([tt.timetuple().tm_yday for tt in times])
    mean = np.log(1.55) + 0.42 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    x = np.empty(n)
    x[0] = 0.0
    phi = 0.985
    e = rng.normal(0, 0.45 * np.sqrt(1 - phi**2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    hs = np.exp(mean + x)
    hours = np.array([(tt - COMMISSIONED).total_seconds() / 3600 for tt in times])
    for t_pk, h_pk, w in NAMED_STORMS:
        h0 = (t_pk - COMMISSIONED).total_seconds() / 3600
        bump = h_pk * np.exp(-0.5 * ((hours - h0) / (w / 1.5)) ** 2)
        hs = np.maximum(hs, bump)
    return times, hs


def interp_rate(table_hs, table_d, hs):
    """Log-log interpolation of damage rate vs Hs (linear extrapolation in log)."""
    lh = np.log(table_hs)
    ld = np.log(np.maximum(table_d, 1e-300))
    x = np.log(np.maximum(hs, 0.2))
    slope_lo = (ld[1] - ld[0]) / (lh[1] - lh[0])
    slope_hi = (ld[-1] - ld[-2]) / (lh[-1] - lh[-2])
    y = np.interp(x, lh, ld)
    y = np.where(x < lh[0], ld[0] + slope_lo * (x - lh[0]), y)
    y = np.where(x > lh[-1], ld[-1] + slope_hi * (x - lh[-1]), y)
    return np.exp(y)


def r2(x, n=4):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


if __name__ == "__main__":
    with open(os.path.join(HERE, "demo_data.json")) as fh:
        data = json.load(fh)
    specs = [t["spec"] for t in data["turbines"]]
    jobs = [(s, hs) for s in specs for hs in HS_BINS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rates = list(pool.map(damage_rates, jobs))
    table, stats, props = {}, {}, {}
    for name, hs, out, st, pr in rates:
        table.setdefault(name, {})[hs] = out
        stats.setdefault(name, {})[hs] = st
        props[name] = pr

    times, hs_ref = hs_history()
    years = np.array([(tt - COMMISSIONED).total_seconds() / (365.25 * 86400) for tt in times])
    storm_years = (STORM_PEAK - COMMISSIONED).total_seconds() / (365.25 * 86400)
    week = 24 * 7
    idx = np.arange(0, times.size, week)
    dff = bd.DFF

    # storms at the reference point (peaks above 6 m)
    storms = []
    for t_pk, h_pk, _ in NAMED_STORMS:
        storms.append(dict(date=t_pk.strftime("%Y-%m-%d"), hs=h_pk))
    storms.append(dict(date=STORM_PEAK.strftime("%Y-%m-%d"), hs=REF_HS, current=True))

    for t in data["turbines"]:
        name, spec = t["spec"]["name"], t["spec"]
        expo = spec["Hs"] / REF_HS
        hs_loc = hs_ref * expo
        hist = {}
        for k in HOT:
            d_tab = np.array([table[name][h][k] for h in HS_BINS])
            rate = interp_rate(np.array(HS_BINS), d_tab, hs_loc)          # per hour
            cum = np.cumsum(rate)
            used = cum * dff[k] * 100                                    # % of design life
            storm_now = t["final"][k]["damage"] * dff[k] * 100          # this storm, %
            total = used[-1] + storm_now
            # damage of each named storm: hours within +-2 half-widths of its peak
            per_storm = []
            for (t_pk, h_pk, w) in NAMED_STORMS:
                h0 = (t_pk - COMMISSIONED).total_seconds() / 3600
                sel = (np.arange(times.size) > h0 - 2 * w) & (np.arange(times.size) < h0 + 2 * w)
                per_storm.append(float(rate[sel].sum() * dff[k] * 25 * 365))   # design-life days
            per_storm.append(t["final"][k]["days"])
            hist[k] = dict(used=r2(np.append(used[idx], total), 3), total=float(total),
                           storm_days=r2(per_storm, 1),
                           projected_life=float(storm_years / max(total / 100, 1e-9)),
                           rates=r2(d_tab, 12))
        tb = Turbine(depth=spec["depth"], scour=spec["scour"], preload_frac=spec["preload"])
        mdl = Model(tb)
        z_mw = props[name]["z_mw"]
        from structure import section, bolt_force
        A_d, _, W_d = section(tb.diameter(tb.z_door), tb.thickness(tb.z_door))
        A_p, _, W_p = section(tb.D_pile, tb.t_pile)
        _, bp = bolt_force(tb, 0.0)
        sec = {k: {"mean": [stats[name][h][k]["mean"] for h in HS_BINS],
                   "std": [stats[name][h][k]["std"] for h in HS_BINS],
                   "nu": [stats[name][h][k]["nu"] for h in HS_BINS]} for k in ("flange", "door", "mudline")}
        t["history"] = dict(years=r2(np.append(years[idx], storm_years), 4), hot=hist,
                            exposure=round(expo, 3))
        t["whatif"] = dict(
            sections=sec, f1=props[name]["f1"],
            z=dict(flange=tb.z_interface, door=tb.z_door, mudline=z_mw, hub=tb.z_hub),
            door=dict(W=W_d, A=A_d, N=mdl.weight_above(tb.z_door), t=float(tb.thickness(tb.z_door))),
            mudline=dict(W=W_p, A=A_p, N=mdl.weight_above(z_mw), t=tb.t_pile),
            flange=dict(N=mdl.weight_above(tb.z_interface), n=tb.n_bolts, Dbc=tb.D_bolt_circle,
                        As=tb.bolt_As, d=tb.bolt_d, **bp, p=0.12, lam=2.1),
            base_thrust=tb.mean_thrust)
        worst = max(HOT, key=lambda k: hist[k]["total"])
        print(f"{name}: life used {hist[worst]['total']:.1f}% at {worst} after "
              f"{storm_years:.1f} y (design pace {100 * storm_years / 25:.1f}%), "
              f"projected life {hist[worst]['projected_life']:.0f} y", flush=True)

    daily = np.arange(0, times.size, 24)
    data["history"] = dict(
        commissioned=COMMISSIONED.strftime("%Y-%m-%d"), storm_peak=STORM_PEAK.strftime("%Y-%m-%d"),
        years_in_service=storm_years, hs_bins=HS_BINS, tp_bins=[tp_of(h) for h in HS_BINS],
        storms=storms,
        days=r2(years[daily], 4), hs_mean=float(hs_ref.mean()),
        hs_daily_max=r2([hs_ref[i:i + 24].max() for i in daily], 2))
    with open(os.path.join(HERE, "demo_data.json"), "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("updated demo_data.json", os.path.getsize(os.path.join(HERE, "demo_data.json")) // 1024, "kB")
