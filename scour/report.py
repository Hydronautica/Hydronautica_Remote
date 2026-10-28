"""Scour assessment report: thresholds, development in time, alert dates, consequences."""

import numpy as np

from .core import AMPLIFICATION, armour_utilisation, assess, onset_current, simulate
from .metocean import to_datetime


def growth_events(time, S, hs, uc, min_rise=0.05, gap_hours=48):
    """Episodes in which the scour depth grew, ranked by the increase (m)."""
    rise = np.diff(S, prepend=S[0]) > 1e-4
    idx = np.flatnonzero(rise)
    if not len(idx):
        return []
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap_hours:
            groups.append((start, prev)); start = i
        prev = i
    groups.append((start, prev))
    ev = []
    for a, b in groups:
        dS = S[b] - S[max(a - 1, 0)]
        if dS >= min_rise:
            ev.append(dict(start=to_datetime(time[a]), end=to_datetime(time[b]), rise=float(dS),
                           hs_max=float(hs[a:b + 1].max()), uc_max=float(uc[a:b + 1].max())))
    return sorted(ev, key=lambda e: -e["rise"])


def analyse(pile, rec, alerts=(0.25, 0.5, 1.0), allowance=0.5, armour_d50=None, s0=0.0):
    """Run the scour model through a record and summarise it.

    alerts and allowance are fractions of the pile diameter; allowance is the scour
    depth assumed in design (above it the foundation is outside its design basis).
    """
    dt = float(np.diff(rec["time"][:2]).astype("timedelta64[s]").astype(float)[0]) / 3600 if len(rec["time"]) > 1 else 1.0
    S, st = simulate(pile, dt, rec["uc"], rec["hs"], rec["tp"], s0=s0)
    sed, D = pile.sediment, pile.D
    t = rec["time"]
    years = (t - t[0]).astype("timedelta64[s]").astype(float) / (365.25 * 86400)
    storm_hs = float(np.quantile(rec["hs"], 0.99))
    storm_tp = float(np.median(rec["tp"][rec["hs"] >= storm_hs]))
    thresholds = dict(
        onset_calm=onset_current(pile, 0.0, 8.0, "onset"), live_calm=onset_current(pile, 0.0, 8.0, "live"),
        onset_storm=onset_current(pile, storm_hs, storm_tp, "onset"), live_storm=onset_current(pile, storm_hs, storm_tp, "live"),
        storm_hs=storm_hs, storm_tp=storm_tp)
    reach = {}
    for a in list(alerts) + [allowance]:
        hit = np.flatnonzero(S >= a * D)
        reach[a] = None if not len(hit) else dict(date=to_datetime(t[hit[0]]), years=float(years[hit[0]]))
    one_year = np.searchsorted(years, 1.0)
    out = dict(
        pile=dict(D=D, h=pile.h), sediment=dict(d50=sed.d50, d_star=sed.d_star, theta_cr=sed.theta_cr, tau_cr=sed.tau_cr),
        thresholds=thresholds, frac_onset=float(st["onset"].mean()), frac_live=float(st["live"].mean()),
        uc_p50=float(np.median(rec["uc"])), uc_p99=float(np.quantile(rec["uc"], 0.99)),
        s_end=float(S[-1]), s_max=float(S.max()), s_1y=float(S[min(one_year, len(S) - 1)]),
        s_eq_max=float(st["s_eq"].max()), s_eq_p95=float(np.quantile(st["s_eq"], 0.95)),
        reach=reach, allowance=allowance, alerts=list(alerts),
        events=growth_events(t, S, rec["hs"], rec["uc"])[:6], S=S, state=st, years=years,
        record=dict(start=to_datetime(t[0]), end=to_datetime(t[-1]), years=float(years[-1])))
    if armour_d50:
        u = armour_utilisation(armour_d50, rec["uc"], rec["hs"], rec["tp"], pile.h)
        i = int(np.argmax(u))
        out["armour"] = dict(d50=armour_d50, u_max=float(u.max()), date=to_datetime(t[i]), hs=float(rec["hs"][i]),
                             uc=float(rec["uc"][i]), frac_over_08=float((u > 0.8).mean()), frac_over_1=float((u > 1).mean()))
    out["level"], out["why"] = verdict(out)
    return out


def verdict(r):
    """Red / amber / green: when to worry about scour at this location."""
    D, allow = r["pile"]["D"], r["allowance"]
    hit = r["reach"].get(allow)
    if hit:
        return "red", (f"Unprotected scour reaches the design allowance ({allow:g}D = {allow * D:.1f} m) after "
                       f"{hit['years'] * 12:.0f} months ({hit['date']:%Y-%m-%d}). Scour protection or a monitoring and "
                       f"remediation plan is needed from installation.")
    if r["s_max"] >= 0.5 * allow * D:
        return "amber", (f"Scour grows to {r['s_max']:.1f} m ({r['s_max'] / D:.2f}D), over half the design allowance. "
                         "Survey after the events listed and after each winter.")
    if r["frac_onset"] < 0.01:
        return "green", "Scour onset is rare at this site; routine surveys are enough."
    return "green", f"Scour stays below {r['s_max']:.1f} m ({r['s_max'] / D:.2f}D); routine surveys are enough."


def frequency_consequence(turbine_kwargs, depths):
    """First natural frequency against scour depth from the demo global model (optional)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "stormtwin"))
    from structure import Model, Turbine
    return [float(Model(Turbine(**dict(turbine_kwargs, scour=float(s)))).frequencies(n=1)[0]) for s in depths]


def text(r, name="site"):
    D = r["pile"]["D"]
    th, sd = r["thresholds"], r["sediment"]
    v = lambda u: f"{u:.2f} m/s" if u > 0 else "waves alone suffice"
    L = [f"Scour assessment: {name}",
         f"  Pile D {D:g} m in {r['pile']['h']:.1f} m of water; sand d50 {sd['d50'] * 1e3:.2f} mm "
         f"(D* {sd['d_star']:.1f}, theta_cr {sd['theta_cr']:.3f}, tau_cr {sd['tau_cr']:.2f} Pa)",
         f"  Record {r['record']['start']:%Y-%m-%d} to {r['record']['end']:%Y-%m-%d} ({r['record']['years']:.1f} y); "
         f"current median {r['uc_p50']:.2f} m/s, 99% {r['uc_p99']:.2f} m/s; storm sea (99% Hs) {th['storm_hs']:.1f} m, Tp {th['storm_tp']:.1f} s",
         "",
         "When scour starts (depth-averaged current)",
         f"  calm sea:   onset at the pile {v(th['onset_calm'])}, whole bed mobile {v(th['live_calm'])}",
         f"  storm sea:  onset at the pile {v(th['onset_storm'])}, whole bed mobile {v(th['live_storm'])}",
         f"  scour active at the pile {100 * r['frac_onset']:.0f}% of the time, live-bed {100 * r['frac_live']:.0f}%",
         "",
         "How bad it gets (no scour protection)",
         f"  equilibrium depth: up to {r['s_eq_max']:.1f} m ({r['s_eq_max'] / D:.2f}D), 95% of the time below {r['s_eq_p95']:.1f} m",
         f"  scour after 1 year {r['s_1y']:.1f} m, maximum {r['s_max']:.1f} m ({r['s_max'] / D:.2f}D), at the end {r['s_end']:.1f} m"]
    for a in r["alerts"] + ([r["allowance"]] if r["allowance"] not in r["alerts"] else []):
        hit = r["reach"][a]
        tag = " (design allowance)" if a == r["allowance"] else ""
        L.append(f"  reaches {a:g}D = {a * D:.1f} m{tag}: " + (f"{hit['date']:%Y-%m-%d}, after {hit['years'] * 12:.0f} months" if hit else "not within the record"))
    if r["events"]:
        L += ["", "Main growth episodes"]
        for e in r["events"]:
            L.append(f"  {e['start']:%Y-%m-%d} to {e['end']:%Y-%m-%d}: +{e['rise']:.2f} m (Hs up to {e['hs_max']:.1f} m, current up to {e['uc_max']:.2f} m/s)")
    if "armour" in r:
        a = r["armour"]
        L += ["", f"Rock protection d50 {a['d50'] * 1e3:.0f} mm",
              f"  peak utilisation {a['u_max']:.2f} on {a['date']:%Y-%m-%d} (Hs {a['hs']:.1f} m, current {a['uc']:.2f} m/s); "
              f"above 0.8 for {100 * a['frac_over_08']:.2f}% of the time, above 1 for {100 * a['frac_over_1']:.2f}%"]
    if "f1" in r:
        L += ["", "Consequence for the structure (demo global model)"]
        L += [f"  scour {s:4.1f} m: f1 {f:.4f} Hz ({100 * (f / r['f1'][0][1] - 1):+.1f}%)" for s, f in r["f1"]]
    L += ["", f"Verdict: {r['level'].upper()}. {r['why']}",
          f"  (onset uses {AMPLIFICATION:g}x bed-shear amplification at the pile; equilibrium after Sumer & Fredsøe with a clear-water reduction)"]
    return "\n".join(L)


def plot(r, rec, pile, path, name="site", tp_coef=3.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pile.D
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.25)
    yrs = r["years"]
    # daily maxima keep multi-year records readable (and avoid aliasing the tide)
    step = max(1, int(round(24 / max(1e-9, (yrs[1] - yrs[0]) * 8766)))) if len(yrs) > 1 else 1
    nd = len(yrs) // step * step
    daily = lambda x: np.asarray(x)[:nd].reshape(-1, step).max(axis=1)
    yd = np.asarray(yrs)[:nd:step]
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(yd, daily(rec["hs"]), lw=0.6, color="#2d7fa3")
    ax.set_ylabel("daily max Hs (m)"); ax.set_title(f"{name}: metocean and scour development", loc="left")
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(yd, daily(rec["uc"]), lw=0.5, color="#566676")
    for k, c, l in (("onset_calm", "#b5730a", "onset at pile (calm)"), ("live_calm", "#bb3328", "bed mobile (calm)")):
        ax2.axhline(r["thresholds"][k], color=c, ls="--", lw=1, label=l)
    ax2.set_ylabel("daily max current (m/s)"); ax2.legend(loc="upper right", fontsize=8)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
    wk = 7 * step
    nw = len(yrs) // wk * wk
    if nw:
        ywk = np.asarray(yrs)[:nw:wk]
        seq_w = np.median(np.asarray(r["state"]["s_eq"])[:nw].reshape(-1, wk), axis=1) / D
        ax3.fill_between(ywk, 0, seq_w, step="post", color="#9fb6c4", alpha=0.35, lw=0, label="equilibrium depth, weekly median")
    ax3.plot(yrs, r["S"] / D, lw=2, color="#0a6a8a", label="scour depth")
    for a in r["alerts"]:
        ax3.axhline(a, color="#888", ls=":", lw=1)
        ax3.text(0.005, a, f"{a:g}D ", transform=ax3.get_yaxis_transform(), va="bottom", ha="left", fontsize=8, color="#666")
    ax3.axhline(r["allowance"], color="#bb3328", ls="--", lw=1.2, label=f"design allowance {r['allowance']:g}D")
    ax3.set_ylabel("S / D"); ax3.set_xlabel("years since start of record"); ax3.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax3.set_ylim(0, max(1.4, r["s_eq_max"] / D * 1.05))
    # regime map
    axm = fig.add_subplot(gs[:, 1])
    uc = np.linspace(0, max(1.2, r["uc_p99"] * 1.3), 70)
    hs = np.linspace(0, max(6, float(np.max(rec["hs"])) * 1.05), 60)
    U, H = np.meshgrid(uc, hs)
    st = assess(pile, U, H, np.maximum(4.0, tp_coef * np.sqrt(H)))
    cf = axm.contourf(U, H, st["s_eq"] / D, levels=np.linspace(0, 1.4, 15), cmap="Blues")
    axm.contour(U, H, 4 * st["theta"] / pile.sediment.theta_cr, levels=[1], colors="#b5730a", linewidths=1.5)
    axm.contour(U, H, st["theta"] / pile.sediment.theta_cr, levels=[1], colors="#bb3328", linewidths=1.5)
    k = np.random.default_rng(0).choice(len(rec["hs"]), size=min(4000, len(rec["hs"])), replace=False)
    axm.scatter(rec["uc"][k], rec["hs"][k], s=2, color="#0e1b27", alpha=0.25, lw=0)
    fig.colorbar(cf, ax=axm, label="equilibrium S / D", shrink=0.6)
    axm.set_xlabel("depth-averaged current (m/s)"); axm.set_ylabel("Hs (m)")
    axm.set_title("Scour regime (dots: record)\norange: onset at pile, red: bed mobile", loc="left", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
