"""Run the StormTwin demo chain on a synthetic storm and write demo_data_<farm>.json.

The farm (layout, turbine type, storm, conditions) comes from farms.py, chosen
with the environment variable STORMTWIN_FARM (A, B or C; default A).
For each turbine:
  1. Sea: NLS-evolved random sea at the turbine (20 min at the storm peak).
  2. Rogue detection: zero-crossing analysis of the 2nd-order surface record.
  3. Screening: narrow-band kinematics (Wheeler) -> Morison -> nonlinear global
     model over the full record -> hot-spot stresses -> fatigue (rainflow).
  4. Deep-dive (flagged turbines): 2nd-order NLS reconstruction of the event,
     slamming check, global model re-run over the event window -> ULS.
  5. Verdict: red / amber / green from pre-agreed rules, with actions.

Everything is synthetic: the storm, the farm and the turbine are illustrative.
Run:  STORMTWIN_FARM=B python build_data.py   (a few minutes, 4 processes)
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per worker process (the farm runs 4 processes in parallel).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import farms  # noqa: E402
import nls  # noqa: E402
from structure import (G, Model, Turbine, bolt_damage, bolt_force, miner_damage,  # noqa: E402
                       morison_nodal, rainflow, section, segment_force, slam_nodal)

CFG = farms.current()
TURB = farms.TURBINES[CFG["turbine"]]
OUT = os.path.join(HERE, f"demo_data_{CFG['id']}.json")
TP = CFG["Tp"]           # spectral peak / carrier period (s)
REL_BANDWIDTH = 0.30     # Gaussian spectral width in k, relative to k0 (storm sea)
RECORD = float(os.environ.get("STORMTWIN_RECORD", 1200.0))   # s of sea per run
DT = 0.05                # structural time step (s)
STORM_PEAK_HOURS = 6.0   # duration the 20-min record is taken to represent
DESIGN_LIFE = 25.0       # years
DFF = {"flange_bolts": 2.0, "door_frame": 2.0, "mudline_weld": 3.0}
SCF = {"door_frame": 2.5, "mudline_weld": 1.2}         # fatigue hot-spot factors
SCF_ULS = {"door_frame": 1.5, "mudline_weld": 1.0}     # net-section factors for ULS

FARM = CFG["farm"]
CONDITION = CFG["condition"]


def zero_crossing(t, eta):
    up = np.flatnonzero((eta[:-1] < 0) & (eta[1:] >= 0))
    waves = []
    for a, b in zip(up[:-1], up[1:]):
        seg = eta[a:b + 1]
        waves.append((t[a], seg.max() - seg.min(), seg.max(), t[a + np.argmax(seg)]))
    return waves


def nls_sea(spec, Tp=TP):
    dw = nls.DeepWater(spec["Hs"] / (2 * np.sqrt(2)), period=Tp)
    kw = REL_BANDWIDTH / (np.sqrt(2) * dw.eps)
    solver = nls.NLSSolver(L=200, N=4096)
    psi0 = nls.random_wave_field(solver.x, k_width=kw, mean_amplitude=1.0, seed=spec["seed"])
    spin = solver.run(psi0, 0.0, 2.0, dt=0.002, n_out=2)
    T0 = 2.0
    T1 = T0 + dw.to_normalised_t(RECORD)
    res = solver.run(spin.psi[-1], T0, T1, dt=0.002, n_out=int(RECORD) + 1)
    return dw, solver, res


def envelope_at(dw, res, x=0.0):
    """Physical envelope A(t) at fixed position x from the NLS snapshots."""
    K = 2 * np.pi * np.fft.fftfreq(res.x.size, d=res.x[1] - res.x[0])
    t = dw.to_physical_t(res.t)
    X = dw.to_normalised_x(x, t)
    psi_hat = np.fft.fft(res.psi, axis=1) / res.x.size
    vals = np.sum(psi_hat * np.exp(1j * (X[:, None] - res.x[0]) * K[None, :]), axis=1)
    return t, dw.envelope(vals)


def screening_kinematics(tb, model, t, A, At, dw, phase=0.0):
    w0 = dw.w0
    th = -w0 * t + phase
    eta = np.real(A * np.exp(1j * th)) + np.real(0.5 * dw.k0 * A**2 * np.exp(2j * th))
    k = float(nls.wavenumber(w0, tb.depth))
    z = model.z[model.wet]
    d = tb.depth
    zs = d * (z[None, :] - eta[:, None]) / (d + eta[:, None])          # Wheeler
    Ch = np.cosh(k * (zs + d)) / np.sinh(k * d)
    ph = np.exp(1j * th)[:, None]
    u = np.real(w0 * A[:, None] * Ch * ph)
    ax = np.real(w0 * (At - 1j * w0 * A)[:, None] * Ch * ph)
    return eta, u, ax, k


def hot_spots(tb, model, U, z_mud_weld):
    """Stress / force histories at the hot spots."""
    out = {}
    N_fl = model.weight_above(tb.z_interface)
    M_fl = model.moment(U, tb.z_interface)
    Z = segment_force(tb, M_fl, N_fl)
    Fb, bp = bolt_force(tb, Z)
    out["flange"] = dict(M=M_fl, Z=Z, Fb=Fb, params=bp)
    D, t = tb.diameter(tb.z_door), tb.thickness(tb.z_door)
    A, I, W = section(D, t)
    M_door = model.moment(U, tb.z_door)
    sN = model.weight_above(tb.z_door) / A
    out["door"] = dict(M=M_door, s=SCF["door_frame"] * M_door / W - sN,
                       s_uls=SCF_ULS["door_frame"] * M_door / W - sN, t=t)
    A, I, W = section(tb.D_pile, tb.t_pile)
    M_mw = model.moment(U, z_mud_weld)
    sN = model.weight_above(z_mud_weld) / A
    out["mudline"] = dict(M=M_mw, s=SCF["mudline_weld"] * M_mw / W - sN,
                          s_uls=SCF_ULS["mudline_weld"] * M_mw / W - sN, t=tb.t_pile)
    out["rot"] = np.degrees(U[:, 2 * model.node(-tb.depth) + 1])
    return out


def bolt_uls(tb, Z, lam=2.1):
    """Flange ULS: open-joint bolt force lam * Z (prying) against the bolt
    tension resistance 0.9 f_ub A_s / 1.25 (EN 1993-1-8); preload not added."""
    return float(lam * np.max(Z) / (0.9 * tb.bolt_fub * tb.bolt_As / 1.25))


def assess(tb, model, hs, scale):
    """Fatigue (scaled to the storm peak duration) and ULS from hot-spot histories."""
    res = {}
    fl = hs["flange"]
    r, c = rainflow(fl["Fb"] / tb.bolt_As)
    Db = bolt_damage(r, c, tb.bolt_d) * scale
    res["flange_bolts"] = dict(
        uls=bolt_uls(tb, fl["Z"]), damage=Db,
        days=Db * DFF["flange_bolts"] * DESIGN_LIFE * 365,
        gap_opening=int(np.sum((fl["Z"][1:] > fl["params"]["Z1"]) & (fl["Z"][:-1] <= fl["params"]["Z1"]))),
        full_open=bool(np.max(fl["Z"]) > fl["params"]["Z2"]))
    fyd = tb.fy / 1.1
    for key, name, curve in (("door", "door_frame", "D_air"),
                             ("mudline", "mudline_weld", "D_seawater_cp")):
        s = hs[key]["s"]
        r, c = rainflow(s)
        Dm = miner_damage(r, c, curve, t=hs[key]["t"]) * scale
        res[name] = dict(uls=float(np.max(np.abs(hs[key]["s_uls"])) / fyd), damage=Dm,
                         days=Dm * DFF[name] * DESIGN_LIFE * 365)
    res["foundation"] = dict(rot_peak=float(np.max(np.abs(hs["rot"]))))
    return res


FATIGUE_DAYS_INSPECT = 60     # storm fatigue that triggers inspection of an accessible hot spot
FATIGUE_DAYS_RBI = 90         # storm fatigue that triggers an RBI update (inaccessible hot spot)


def verdict(name, a, airgap, freq_drop, slam, rogue, preload=1.0):
    """Pre-agreed red / amber / green rules. Returns status, reasons, actions."""
    reasons, actions = [], []
    red = amber = False
    uls = {k: a[k]["uls"] for k in ("flange_bolts", "door_frame", "mudline_weld")}
    labels = {"flange_bolts": "tower-base flange bolts", "door_frame": "door frame",
              "mudline_weld": "mudline weld"}
    for k, u in uls.items():
        if u >= 0.9:
            red = True
            reasons.append(f"ULS utilisation {u:.2f} at {labels[k]} (act at 0.90)")
        elif u >= 0.7:
            amber = True
            reasons.append(f"ULS utilisation {u:.2f} at {labels[k]} (review at 0.70)")
    fb = a["flange_bolts"]
    if fb["full_open"]:
        red = True
        reasons.append("Flange gap fully open during the event (prying)")
    elif fb["gap_opening"] > 0 and preload < 0.9:
        red = True
        reasons.append(f"Flange gap opened {fb['gap_opening']}x with preload already at "
                       f"{preload:.0%}: further preload loss likely")
    elif fb["gap_opening"] > 0:
        amber = True
        reasons.append(f"Flange gap opened {fb['gap_opening']}x in 20 min at the storm peak")
    for k in ("flange_bolts", "door_frame"):
        if a[k]["days"] >= FATIGUE_DAYS_INSPECT:
            amber = True
            reasons.append(f"Storm used {a[k]['days']:.0f} days of design fatigue life at {labels[k]}")
    rbi = a["mudline_weld"]["days"] >= FATIGUE_DAYS_RBI
    if rbi:
        reasons.append(f"Storm used {a['mudline_weld']['days']:.0f} days of design fatigue life at "
                       "the mudline weld (logged to RBI; not inspectable)")
    if a["foundation"]["rot_peak"] >= 0.5:
        red = True
        reasons.append(f"Peak mudline rotation {a['foundation']['rot_peak']:.2f} deg (limit 0.5)")
    if freq_drop >= 3.0:
        amber = True
        reasons.append(f"First natural frequency down {freq_drop:.1f}% at peak response")
    if airgap < 1.5:
        amber = True
        reasons.append(f"Crest within {airgap:.1f} m of the platform underside")
    if slam:
        amber = True
        reasons.append("Breaking-wave slamming on the monopile")
    status = "red" if red else "amber" if amber else "green"
    if fb["gap_opening"] > 0 or fb["days"] >= FATIGUE_DAYS_INSPECT or uls["flange_bolts"] >= 0.7:
        urgency = "before restart" if status == "red" else "at the next campaign"
        actions.append(("Tower-base flange", f"Verify bolt preload on a 10% sample (ultrasonic) {urgency}; re-tension below 90%"))
    if uls["door_frame"] >= 0.7 or a["door_frame"]["days"] >= FATIGUE_DAYS_INSPECT:
        actions.append(("Door frame", "Visual + MPI of the door-frame weld toes at the next campaign"))
    if uls["mudline_weld"] >= 0.7 or rbi:
        actions.append(("Mudline weld", "Update the RBI model with this storm's damage; raise monitoring priority"))
    if freq_drop >= 3.0 or a["foundation"]["rot_peak"] >= 0.25:
        actions.append(("Foundation", "Confirm frequency recovery within 72 h; check inclinometer; schedule scour survey"))
    if airgap < 1.5 or slam:
        actions.append(("Secondary steel", "Inspect boat landing, platform and J-tube clamps for wave-impact damage"))
    if not actions:
        actions.append(("All", "No action. Storm logged; fatigue totals updated"))
    if rogue:
        reasons.append("Rogue wave recorded (H > 2 Hs)")
    return status, reasons, actions


def r2(x, n=3):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


def process(spec):
    import warnings
    warnings.filterwarnings("ignore")
    tb = Turbine(**TURB, depth=spec["depth"], scour=spec["scour"], preload_frac=spec["preload"],
                 name=spec["name"])
    model = Model(tb)
    f_cal = model.setup_damping()
    dw, solver, res = nls_sea(spec)
    t1, A1 = envelope_at(dw, res)
    t1 = t1 - t1[0]
    t = np.arange(0, RECORD + 1e-9, DT)
    csr, csi = CubicSpline(t1, A1.real), CubicSpline(t1, A1.imag)
    A = csr(t) + 1j * csi(t)
    At = csr(t, 1) + 1j * csi(t, 1)
    eta, u, ax, k = screening_kinematics(tb, model, t, A, At, dw)
    c_phase = dw.w0 / k

    # rogue detection
    waves = zero_crossing(t, eta)
    Hs_m = 4 * np.std(eta)
    H = np.array([w[1] for w in waves])
    iw = int(np.argmax(H))
    t_ev, Hmax, Cmax, t_crest = waves[iw][0], H[iw], waves[iw][2], waves[iw][3]
    rogue = bool(Hmax > 2 * Hs_m or Cmax > 1.25 * Hs_m)

    # screening global run
    F = morison_nodal(model, eta, u, ax)
    ramp = np.clip(t / 20.0, 0, 1)
    F *= (0.5 - 0.5 * np.cos(np.pi * ramp))[:, None]
    U, _ = model.run(t, F)
    mz = model.z[model.soil]
    peak_i = int(np.argmax(np.abs(model.moment(U, -tb.depth))))
    Ms = np.array([abs(model.moment(U[peak_i:peak_i + 1], zz)[0]) for zz in mz])
    z_mw = float(mz[int(np.argmax(Ms))])
    hs = hot_spots(tb, model, U, z_mw)
    scale = STORM_PEAK_HOURS * 3600 / RECORD
    a_scr = assess(tb, model, hs, scale)
    f_peak = model.frequencies(U[int(np.argmax(np.abs(U[:, 2 * model.node(-tb.depth)])))], n=1)[0]
    freq_drop = 100 * (1 - f_peak / f_cal[0])
    airgap = tb.z_platform - float(eta.max())

    status, reasons, actions = verdict(spec["name"], a_scr, airgap, freq_drop, False, rogue,
                                       spec["preload"])
    flagged = status != "green" or rogue or spec["name"] in CONDITION
    out = dict(spec=spec, condition=CONDITION.get(spec["name"]), f1=float(f_cal[0]),
               Hs=float(Hs_m), Hmax=float(Hmax), Cmax=float(Cmax), t_event=float(t_crest),
               rogue=rogue, n_waves=len(waves), airgap=airgap, freq_drop=freq_drop,
               z_mudline_weld=z_mw, screening=a_scr)

    # downsampled record for display
    sel = slice(None, None, 5)                               # 0.25 s
    out["record"] = dict(t=r2(t[sel], 2), eta=r2(eta[sel], 2))
    out["screen_series"] = dict(M_flange=r2(hs["flange"]["M"][::20] / 1e6, 1),
                                t=r2(t[::20], 1))

    deep = None
    if flagged:
        deep = deep_dive(spec, tb, model, dw, solver, res, t, U, t_crest, c_phase, scale, hs, A)
        a_use = dict(a_scr)
        for kk in ("flange_bolts", "door_frame", "mudline_weld"):
            a_use[kk] = dict(a_scr[kk], uls=max(a_scr[kk]["uls"], deep["uls"][kk]))
        a_use["flange_bolts"]["full_open"] = a_scr["flange_bolts"]["full_open"] or deep["full_open"]
        a_use["foundation"] = dict(rot_peak=max(a_scr["foundation"]["rot_peak"], deep["rot_peak"]))
        status, reasons, actions = verdict(spec["name"], a_use, min(airgap, deep["airgap"]),
                                           freq_drop, deep["slam"], rogue, spec["preload"])
        out["final"] = a_use
    else:
        out["final"] = a_scr
    out["deep"] = deep
    out["status"], out["reasons"], out["actions"] = status, reasons, actions
    print(f"{spec['name']}: {status:5s} Hs={Hs_m:.2f} Hmax={Hmax:.2f} ({Hmax / Hs_m:.2f} Hs) "
          f"f1={f_cal[0]:.3f} deep={'yes' if deep else 'no'}", flush=True)
    return out


def deep_dive(spec, tb, model, dw, solver, res, t, U, t_crest, c_phase, scale, hs_scr, A_scr):
    """2nd-order NLS reconstruction of the event and a global re-run over the window."""
    half = 40.0
    tw0, tw1 = max(t_crest - half, 25.0), min(t_crest + half, RECORD)
    # NLS snapshots are 1 s apart; restart from the one before the window
    t_snap = dw.to_physical_t(res.t) - dw.to_physical_t(res.t[0])
    j0 = int(np.searchsorted(t_snap, tw0)) - 1
    j0 = max(j0, 0)
    T_start = res.t[j0]
    T_a = res.t[0] + dw.to_normalised_t(tw0)
    T_b = res.t[0] + dw.to_normalised_t(tw1)
    pre = solver.run(res.psi[j0], T_start, T_a, dt=0.0005, n_out=2) if T_a > T_start else None
    psi_a = pre.psi[-1] if pre is not None else res.psi[j0]
    n = int(round((tw1 - tw0) / DT)) + 1
    fine = solver.run(psi_a, T_a, T_b, dt=0.0005, n_out=n)
    wave = nls.NLSWave(fine, dw, depth=tb.depth, phase=0.0, order=2)
    tw = wave.physical_times() - dw.to_physical_t(res.t[0])
    zw = model.z[model.wet]
    eta = np.empty(tw.size)
    u = np.empty((tw.size, zw.size))
    ax = np.empty_like(u)
    for i, ti in enumerate(wave.physical_times()):
        f = wave.field(ti)
        e = float(f.eta(np.array([0.0]))[0])
        kin = f.kinematics(np.zeros(zw.size), zw, "linear")
        eta[i] = e
        u[i] = np.nan_to_num(kin["u"])
        ax[i] = np.nan_to_num(kin["ut"])
    ic = int(np.argmax(eta))
    kin_c = wave.field(wave.physical_times()[ic]).kinematics(
        np.zeros(60), np.linspace(-tb.depth, eta[ic] - 1e-3, 60), "linear")
    u_crest = float(kin_c["u"][np.isfinite(kin_c["u"])][-1])
    slam = u_crest / c_phase > 0.8
    F = morison_nodal(model, eta, u, ax)
    if slam:
        F += slam_nodal(model, tw, tw[ic], eta[ic], c_phase)
    # start from the screening state at the window start
    i0 = int(round(tw0 / DT))
    v0 = (U[i0 + 1] - U[i0 - 1]) / (2 * DT)
    U2, _ = model.run(tw, F, u0=U[i0], v0=v0)
    hs = hot_spots(tb, model, U2, model_mw(model, U2, tb))
    fyd = tb.fy / 1.1
    uls = dict(flange_bolts=bolt_uls(tb, hs["flange"]["Z"]),
               door_frame=float(np.max(np.abs(hs["door"]["s_uls"])) / fyd),
               mudline_weld=float(np.max(np.abs(hs["mudline"]["s_uls"])) / fyd))
    # screening (narrow-band, Wheeler) profile at the same instant, for comparison
    sw = slice(i0, i0 + tw.size)
    j = i0 + ic
    zc = np.linspace(-tb.depth, eta[ic] - 1e-3, 60)
    A_j, th = A_scr[j], -dw.w0 * t[j]
    k = float(nls.wavenumber(dw.w0, tb.depth))
    e_s = float(np.real(A_j * np.exp(1j * th)) + np.real(0.5 * dw.k0 * A_j**2 * np.exp(2j * th)))
    zs = tb.depth * (zc - e_s) / (tb.depth + e_s)
    u_s = np.real(dw.w0 * A_j * np.cosh(k * (zs + tb.depth)) / np.sinh(k * tb.depth) * np.exp(1j * th))
    u_s = np.where(zc <= e_s, u_s, np.nan)
    return dict(
        screen_profile=dict(z=r2(zc, 2), u=[None if not np.isfinite(v) else round(float(v), 3) for v in u_s]),
        t=r2(tw[::2], 2), eta=r2(eta[::2], 2),
        M_flange=r2(hs["flange"]["M"][::2] / 1e6, 1),
        M_flange_screen=r2(hs_scr["flange"]["M"][sw][::2] / 1e6, 1),
        M_mud=r2(model.moment(U2, -tb.depth)[::2] / 1e6, 1),
        Z=r2(hs["flange"]["Z"][::2] / 1e6, 4), Fb=r2(hs["flange"]["Fb"][::2] / 1e6, 4),
        bolt_params={k: v / 1e6 for k, v in hs["flange"]["params"].items()},
        crest_profile=dict(z=r2(np.linspace(-tb.depth, eta[ic] - 1e-3, 60), 2),
                           u=r2(np.nan_to_num(kin_c["u"]), 3)),
        u_crest=u_crest, c=float(c_phase), slam=bool(slam), crest=float(eta[ic]),
        airgap=tb.z_platform - float(eta.max()),
        uls=uls, full_open=bool(np.max(hs["flange"]["Z"]) > hs["flange"]["params"]["Z2"]),
        rot_peak=float(np.max(np.abs(hs["rot"]))))


def model_mw(model, U, tb):
    mz = model.z[model.soil]
    peak_i = int(np.argmax(np.abs(model.moment(U, -tb.depth))))
    Ms = np.array([abs(model.moment(U[peak_i:peak_i + 1], zz)[0]) for zz in mz])
    return float(mz[int(np.argmax(Ms))])


def bolt_curve(tb):
    Z = np.linspace(-0.3e6, 2.0e6, 120)
    F, p = bolt_force(tb, Z)
    return dict(Z=r2(Z / 1e6, 4), F=r2(F / 1e6, 4))


if __name__ == "__main__":
    only = sys.argv[1:]
    specs = [s for s in FARM if not only or s["name"] in only]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process, specs))
    path = OUT
    if only and os.path.exists(path):
        # re-run of selected turbines: merge into the existing farm results
        with open(path) as fh:
            old = {t["spec"]["name"]: t for t in json.load(fh)["turbines"]}
        old.update({t["spec"]["name"]: t for t in results})
        results = [old[s["name"]] for s in FARM if s["name"] in old]
    tb_ref = Turbine(**TURB)
    data = dict(
        meta=dict(Tp=TP, record_s=RECORD, storm_peak_hours=STORM_PEAK_HOURS,
                  design_life=DESIGN_LIFE, dff=DFF, scf=SCF, scf_uls=SCF_ULS,
                  fatigue_days_inspect=FATIGUE_DAYS_INSPECT, fatigue_days_rbi=FATIGUE_DAYS_RBI,
                  bandwidth=REL_BANDWIDTH,
                  farm=dict(id=CFG["id"], name=CFG["name"], region=CFG["region"],
                            storm_name=CFG["storm_name"],
                            storm_peak=CFG["storm_peak"].strftime("%Y-%m-%d"),
                            commissioned=CFG["commissioned"].strftime("%Y-%m-%d")),
                  turbine=dict(D_pile=tb_ref.D_pile, z_hub=tb_ref.z_hub, m_rna=tb_ref.m_rna,
                               n_bolts=tb_ref.n_bolts, bolt=f"M{round(tb_ref.bolt_d * 1000)} 10.9",
                               z_interface=tb_ref.z_interface, z_door=tb_ref.z_door,
                               z_platform=tb_ref.z_platform, D_tower_top=tb_ref.D_tower_top,
                               t_tower_base=tb_ref.t_tower_base, t_tower_top=tb_ref.t_tower_top,
                               t_pile=tb_ref.t_pile, embedment=tb_ref.embedment,
                               rotor_D=tb_ref.rotor_D, rated_thrust=tb_ref.rated_thrust,
                               rated_U=tb_ref.rated_U, idle_thrust=tb_ref.mean_thrust,
                               label=tb_ref.label)),
        bolt_curve=bolt_curve(tb_ref),
        bolt_curve_low=bolt_curve(Turbine(**TURB, preload_frac=min(f["preload"] for f in FARM))),
        low_preload=min(f["preload"] for f in FARM),
        turbines=results)
    with open(OUT, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("wrote", os.path.basename(OUT), os.path.getsize(OUT) // 1024, "kB")
