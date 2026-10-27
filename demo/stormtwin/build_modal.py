"""Synthetic operational modal tracking for the StormTwin demo.

Generates the natural-frequency history the twin would extract from
accelerometers by operational modal analysis (OMA), for every turbine of the
selected farm, from commissioning to the replayed storm:

  * tower 1st fore-aft / side-side and tower 2nd mode (from the global model,
    re-solved at each scour depth), and the 1st edgewise mode of each blade;
  * environmental and operational variability: temperature, seasonal marine
    growth, operating vs parked, icing (cold sites), soil set-up;
  * events: storm dips with recovery (sized from the storm runs), scour
    growth, bolt-preload loss, blade damage on one blade.

The twin's detection is then applied: a regression on temperature, operating
state and season fitted on the first year gives the expected frequency; the
normalised deviation is checked against +-3 sigma control limits, and blades
are also compared with their siblings. Everything here is synthetic.

Run after build_history.py:  STORMTWIN_FARM=A python build_modal.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import farms  # noqa: E402
from structure import Model, Turbine  # noqa: E402

CFG = farms.current()
TURB = farms.TURBINES[CFG["turbine"]]
OUT = os.path.join(HERE, f"demo_data_{CFG['id']}.json")
STEP_DAYS = 3

BLADE_EDGE = {"15MW": 0.66, "10MW": 0.92}     # nominal 1st edgewise frequency (Hz)

# Damage scenarios (synthetic): (turbine, kind, start date, size)
SCENARIOS = {
    "A": [("A07", "blade_crack", "B", datetime(2024, 5, 10), 0.028),
          ("A02", "preload_loss", None, datetime(2025, 3, 1), 0.012),
          ("A05", "scour", None, None, None)],
    "B": [("B06", "lightning", "C", datetime(2025, 8, 17), 0.025),
          ("B03", "scour", None, None, None)],
    "C": [("C02", "blade_crack", "A", datetime(2023, 4, 1), 0.022),
          ("C04", "preload_loss", None, datetime(2024, 9, 1), 0.012)],
}
COLD_SITE = {"A": False, "B": False, "C": True}
TEMP = {"A": (10.0, 5.0), "B": (14.0, 8.0), "C": (8.0, 9.0)}     # mean, seasonal amplitude (degC)


def days_between(a, b):
    return (b - a).total_seconds() / 86400


def tower_freqs(tb, scour):
    m = Model(Turbine(**dict(TURB, scour=scour), depth=tb.depth, preload_frac=tb.preload_frac))
    f = m.frequencies(n=2)
    return float(f[0]), float(f[1])


def fit_and_flag(t_days, obs, X, base_days=365.0, k=4.0, run=5):
    """Environmental normalisation and control chart.

    Fits obs ~ X on the first year, returns the normalised deviation (%),
    the control limit (%) and alarm indices (``run`` consecutive exceedances).
    """
    base = t_days <= base_days
    coef, *_ = np.linalg.lstsq(X[base], obs[base], rcond=None)
    pred = X @ coef
    dev = 100 * (obs - pred) / pred
    sigma = np.std(dev[base])
    lim = k * sigma
    out = np.abs(dev) > lim
    alarm = np.zeros_like(out)
    c = 0
    for i, o in enumerate(out):
        c = c + 1 if o else 0
        if c >= run:
            alarm[i] = True
    return pred, dev, lim, alarm


def first_alarm(dates, alarm):
    idx = np.flatnonzero(alarm)
    return dates[idx[0]].strftime("%Y-%m-%d") if idx.size else None


if __name__ == "__main__":
    with open(OUT) as fh:
        data = json.load(fh)
    rng = np.random.default_rng(1000 + ord(CFG["id"]))
    start, end = CFG["commissioned"], CFG["storm_peak"]
    n = int(days_between(start, end) // STEP_DAYS) + 1
    dates = np.array([start + timedelta(days=STEP_DAYS * i) for i in range(n)])
    td = np.array([days_between(start, d) for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates])
    season = np.cos(2 * np.pi * (doy - 200) / 365.25)            # +1 in mid-July
    Tm, Ta = TEMP[CFG["id"]]
    temp = Tm + Ta * season + rng.normal(0, 2.0, n)               # daily mean air temperature
    storms = [(s[0], s[1]) for s in CFG["named_storms"]] + [(CFG["storm_peak"], CFG["farm"][0]["Hs"])]
    scen = {s[0]: s for s in SCENARIOS[CFG["id"]]}
    # icing episodes (cold site): a few winter spells, all blades of all turbines
    ice = np.zeros(n)
    if COLD_SITE[CFG["id"]]:
        for y in range(start.year, end.year + 1):
            for _ in range(2):
                d0 = datetime(y, 1, 1) + timedelta(days=int(rng.integers(0, 60)))
                k0 = (td >= days_between(start, d0)) & (td <= days_between(start, d0) + rng.integers(3, 9))
                ice[k0] = 1.0
    alarms_farm = 0
    for t in data["turbines"]:
        spec = t["spec"]
        tb = Turbine(**TURB, depth=spec["depth"], scour=0.0, preload_frac=1.0)
        f1_0, f2_0 = tower_freqs(tb, 0.0)
        name = spec["name"]
        sc = scen.get(name)
        op = (rng.random(n) < 0.82).astype(float)                 # share of time operating each day
        # --- tower 1st mode -------------------------------------------------------
        setup = 0.002 * (1 - np.exp(-td / 300))                   # soil set-up (stiffening)
        growth = -0.003 * (season + 1) / 2                        # marine growth (summer peak)
        oper = 0.004 * (op - 0.82)                                # operating vs parked
        temp_t = -0.00015 * (temp - Tm)
        rel = 1 + setup + growth + oper + temp_t
        # storm dips: size from the storm run, recovering over ~3 days
        drop_now = t.get("freq_drop", 1.5) / 100
        for d_pk, hs_pk in storms:
            tp = days_between(start, d_pk)
            a = drop_now * (hs_pk / storms[-1][1]) ** 2
            after = td >= tp
            rel = rel - np.where(after, a * np.exp(-(td - tp) / 3.0), 0.0)
        events = [dict(date=d.strftime("%Y-%m-%d"), label="storm") for d, _ in storms]
        if sc and sc[1] == "scour":
            s_final = spec["scour"]
            grow = np.clip(td / td[-1], 0, 1) ** 0.6 * s_final
            depths = np.linspace(0, s_final, 6)
            f_s = np.array([tower_freqs(tb, s)[0] for s in depths])
            rel = rel * np.interp(grow, depths, f_s) / f1_0
            events.append(dict(date=dates[-1].strftime("%Y-%m-%d"), label=f"scour survey {s_final} m"))
        if sc and sc[1] == "preload_loss":
            tp = days_between(start, sc[3])
            rel = rel - np.where(td >= tp, sc[4] * np.clip((td - tp) / 180, 0, 1), 0.0)
            events.append(dict(date=sc[3].strftime("%Y-%m-%d"), label="preload loss begins (modelled)"))
        fa = f1_0 * rel + rng.normal(0, 0.0015 * f1_0, n)
        ss = f1_0 * 1.008 * (rel - 0.6 * oper) + rng.normal(0, 0.0015 * f1_0, n)
        f2 = f2_0 * (1 + 0.6 * (rel - 1)) + rng.normal(0, 0.002 * f2_0, n)
        # --- blades (edgewise) -----------------------------------------------------
        fe0 = BLADE_EDGE[CFG["turbine"]] * (1 + rng.normal(0, 0.004))
        blades = {}
        for b in "ABC":
            r = 1 - 0.0005 * (temp - Tm) - 0.03 * ice + rng.normal(0, 0.0012) \
                - 0.002 * np.clip(td / 3650, 0, 1)              # slow ageing
            if sc and sc[1] in ("blade_crack", "lightning") and sc[2] == b:
                tp = days_between(start, sc[3])
                if sc[1] == "blade_crack":
                    r = r - np.where(td >= tp, sc[4] * np.clip((td - tp) / max(td[-1] - tp, 1), 0, 1) ** 1.5, 0.0)
                    lab = "blade crack initiates (synthetic)"
                else:
                    r = r - np.where(td >= tp, sc[4], 0.0)
                    lab = "lightning strike (synthetic)"
                events.append(dict(date=sc[3].strftime("%Y-%m-%d"), label=f"{lab}, blade {b}"))
            blades[b] = fe0 * r + rng.normal(0, 0.003 * fe0, n)
        # --- detection -------------------------------------------------------------
        X = np.column_stack([np.ones(n), temp, op, np.sin(2 * np.pi * doy / 365.25),
                             np.cos(2 * np.pi * doy / 365.25)])
        Xb = np.column_stack([np.ones(n), temp, ice])
        modes = {}
        alarms = []
        for key, obs, XX, label in (("tower_fa", fa, X, "Tower 1st fore-aft"),
                                    ("tower_ss", ss, X, "Tower 1st side-side"),
                                    ("tower_2", f2, X, "Tower 2nd")):
            pred, dev, lim, al = fit_and_flag(td, obs, XX)
            modes[key] = dict(label=label, f=[round(float(v), 5) for v in obs],
                              pred=[round(float(v), 5) for v in pred],
                              dev=[round(float(v), 3) for v in dev], lim=round(float(lim), 3),
                              alarm=first_alarm(dates, al))
            if modes[key]["alarm"]:
                alarms.append(dict(mode=label, date=modes[key]["alarm"]))
        med = np.median(np.vstack([blades[b] for b in "ABC"]), axis=0)
        for b in "ABC":
            pred, dev, lim, al = fit_and_flag(td, blades[b], Xb)
            # blade-to-blade: deviation from the median of the three blades
            rel_b = 100 * (blades[b] - med) / med
            base = td <= 365
            s_b = np.std(rel_b[base])
            al_b = np.abs(rel_b - np.mean(rel_b[base])) > 4 * s_b
            run = np.convolve(al_b.astype(int), np.ones(5, int), mode="full")[:n] >= 5
            label = f"Blade {b} 1st edgewise"
            modes[f"blade_{b}"] = dict(label=label, f=[round(float(v), 5) for v in blades[b]],
                                       pred=[round(float(v), 5) for v in pred],
                                       dev=[round(float(v), 3) for v in dev], lim=round(float(lim), 3),
                                       b2b=[round(float(v), 3) for v in rel_b - np.mean(rel_b[base])],
                                       b2b_lim=round(float(4 * s_b), 3),
                                       alarm=first_alarm(dates, al), alarm_b2b=first_alarm(dates, run))
            if modes[f"blade_{b}"]["alarm_b2b"]:
                alarms.append(dict(mode=label + " (blade-to-blade)", date=modes[f"blade_{b}"]["alarm_b2b"]))
        alarms_farm += len(alarms)
        t["modal"] = dict(modes=modes, events=events, alarms=alarms,
                          f1_design=round(f1_0, 4), f2_design=round(f2_0, 4))
        print(f"{name}: f1 {f1_0:.4f} Hz, f2 {f2_0:.3f} Hz, alarms: "
              + (", ".join(f"{a['mode']} {a['date']}" for a in alarms) or "none"), flush=True)
    data["modal"] = dict(dates=[d.strftime("%Y-%m-%d") for d in dates],
                         years=[round(float(v) / 365.25, 4) for v in td],
                         temp=[round(float(v), 1) for v in temp], ice=[int(v) for v in ice],
                         step_days=STEP_DAYS, cold_site=COLD_SITE[CFG["id"]])
    with open(OUT, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("updated", os.path.basename(OUT), os.path.getsize(OUT) // 1024, "kB;", alarms_farm, "alarms")
