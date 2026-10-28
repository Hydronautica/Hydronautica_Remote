"""Command line: scour estimates from waves and current.

  python -m scour --farm A                      # every turbine of demo farm A
  python -m scour --farm B --turbine B03        # one turbine, with plot and frequency check
  python -m scour --D 8 --h 25 --d50 0.3 --u-spring 0.6 --hs-median 1.2 --years 5
  python -m scour --csv hindcast.csv --D 10 --h 40 --d50 0.25 --armour 0.3

A CSV needs columns time, hs, tp, uc (depth-averaged current speed, m/s) on a
constant step. Without a CSV a synthetic record (seasonal waves, storms,
spring-neap tide, storm surge current) is generated.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from .core import Armour, Pile, Sediment
from .metocean import load_csv, synthetic
from .report import analyse, f1_series, frequency_consequence, plot, plot_protection, text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Seabed and tidal settings for the demo farms (synthetic, illustrative).
SITES = {
    "A": dict(d50=0.30e-3, u_spring=0.65, surge_per_m=0.05, allowance=0.5),
    "B": dict(d50=0.18e-3, u_spring=0.45, surge_per_m=0.06, allowance=0.5),
    "C": dict(d50=0.40e-3, u_spring=0.25, surge_per_m=0.08, allowance=0.5),
}


def farm_config(fid):
    sys.path.insert(0, os.path.join(HERE, "demo", "stormtwin"))
    import farms
    return farms.FARMS[fid](), farms.TURBINES


def armour_of(args, D):
    if not args.armour:
        return None
    return Armour(d50=args.armour, thickness=args.armour_thickness, extent=args.armour_extent)


def run_site(name, pile, rec, args, turbine_kwargs=None, make_plot=True):
    r = analyse(pile, rec, alerts=args.alerts, allowance=args.allowance, armour=armour_of(args, pile.D))
    if turbine_kwargs is None:
        turbine_kwargs = dict(D_pile=pile.D, D_bolt_circle=pile.D - 0.2, depth=pile.h)
    if not args.no_frequency:
        depths = sorted({0.0, round(0.25 * pile.D, 1), round(0.5 * pile.D, 1), round(r["s_max"], 1), round(1.3 * pile.D, 1)})
        r["f1"] = list(zip(depths, frequency_consequence(turbine_kwargs, depths)))
        r["f1t"] = f1_series(turbine_kwargs, r)
    print(text(r, name))
    if make_plot and not args.no_plots:
        os.makedirs(args.out, exist_ok=True)
        path = plot(r, rec, pile, os.path.join(args.out, f"scour_{name}.png"), name=name, tp_coef=args.tp_coef)
        print(f"  plot: {os.path.relpath(path)}")
        if "f1t" in r or "protected" in r:
            path = plot_protection(r, os.path.join(args.out, f"scour_protection_{name}.png"), name=name)
            print(f"  plot: {os.path.relpath(path)}")
    print()
    return r


def summary_row(name, pile, r, f0=None, f1=None):
    D = pile.D
    hit = r["reach"].get(r["allowance"])
    return dict(name=name, depth=pile.h, s_1y=r["s_1y"], s_max=r["s_max"], s_eq_max=r["s_eq_max"],
                frac_onset=r["frac_onset"], allowance_after_months=None if not hit else round(hit["years"] * 12, 1),
                level=r["level"], f1_drop_pct=None if f0 is None else round(100 * (1 - f1 / f0), 2), D=D)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scour", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--farm", choices=sorted(SITES), help="use a demo farm (positions, depths, climate)")
    ap.add_argument("--turbine", help="with --farm: assess one turbine in detail")
    ap.add_argument("--csv", help="metocean record (time, hs, tp, uc)")
    ap.add_argument("--D", type=float, help="pile diameter (m)")
    ap.add_argument("--h", type=float, help="water depth (m)")
    ap.add_argument("--d50", type=float, help="median grain size (mm)")
    ap.add_argument("--u-spring", type=float, default=0.6, help="spring-tide peak depth-averaged current (m/s), synthetic record")
    ap.add_argument("--surge", type=float, default=0.05, help="storm surge current per metre of Hs above the median (m/s/m)")
    ap.add_argument("--hs-median", type=float, default=1.4, help="median Hs (m), synthetic record")
    ap.add_argument("--tp-coef", type=float, default=3.6, help="Tp = coef * sqrt(Hs), synthetic record")
    ap.add_argument("--years", type=float, default=None, help="length of the synthetic record (years)")
    ap.add_argument("--start", default="2024-01-01", help="start date of the synthetic record")
    ap.add_argument("--alerts", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.5, 1.0], help="alert depths as fractions of D")
    ap.add_argument("--allowance", type=float, default=None, help="design scour allowance as a fraction of D (default 0.5)")
    ap.add_argument("--armour", type=float, default=None, help="rock protection d50 (m): run the protected case too")
    ap.add_argument("--armour-thickness", type=float, default=1.5, help="armour plus filter thickness (m)")
    ap.add_argument("--armour-extent", type=float, default=2.5, help="protection rim radius from the pile centre (pile diameters)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "scour"))
    ap.add_argument("--json", help="write the summary to this file")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-frequency", action="store_true", help="skip the natural-frequency consequence check")
    args = ap.parse_args(argv)

    if args.farm:
        cfg, turbines = farm_config(args.farm)
        site = SITES[args.farm]
        if args.allowance is None:
            args.allowance = site["allowance"]
        tb = turbines[cfg["turbine"]]
        sed = Sediment(args.d50 * 1e-3 if args.d50 else site["d50"])
        years = args.years or (cfg["storm_peak"] - cfg["commissioned"]).days / 365.25
        rec = load_csv(args.csv) if args.csv else synthetic(
            cfg["commissioned"].strftime("%Y-%m-%d"), years, hs_median=cfg["climate_median"], seasonal=cfg["seasonal"],
            tp_coef=cfg["tp_coef"], u_spring=site["u_spring"], surge_per_m=site["surge_per_m"],
            storms=cfg["named_storms"], seed=cfg["history_seed"])
        args.tp_coef = cfg["tp_coef"]
        print(f"{cfg['name']} ({cfg['region']}): {len(cfg['farm'])} x {tb['label']}, D {tb['D_pile']:g} m, "
              f"sand d50 {sed.d50 * 1e3:.2f} mm, spring current {site['u_spring']:.2f} m/s, from {cfg['commissioned']:%Y-%m-%d}\n")
        rows, results = [], {}
        chosen = [t for t in cfg["farm"] if not args.turbine or t["name"] == args.turbine]
        if not chosen:
            ap.error(f"no turbine {args.turbine} in farm {args.farm}")
        for t in chosen:
            pile = Pile(D=tb["D_pile"], h=t["depth"], sediment=sed)
            kw = dict(tb, depth=t["depth"])
            if args.turbine:
                r = run_site(t["name"], pile, rec, args, turbine_kwargs=kw)
                results[t["name"]] = r
                continue
            r = analyse(pile, rec, alerts=args.alerts, allowance=args.allowance, armour=armour_of(args, pile.D))
            f0 = f1 = None
            if not args.no_frequency:
                f0, f1 = frequency_consequence(kw, [0.0, r["s_max"]])
            rows.append(summary_row(t["name"], pile, r, f0, f1))
            results[t["name"]] = r
        if rows:
            print(f"{'turbine':8s} {'depth':>6s} {'S 1 y':>7s} {'S max':>7s} {'S/D':>5s} {'onset':>6s} {'to ' + format(args.allowance, 'g') + 'D':>9s} {'f1 drop':>8s}  verdict")
            for w in rows:
                print(f"{w['name']:8s} {w['depth']:5.1f}m {w['s_1y']:6.1f}m {w['s_max']:6.1f}m {w['s_max'] / w['D']:5.2f} {100 * w['frac_onset']:5.0f}% "
                      f"{(str(w['allowance_after_months']) + ' mo') if w['allowance_after_months'] is not None else 'never':>9s} "
                      f"{(format(w['f1_drop_pct'], '.1f') + '%') if w['f1_drop_pct'] is not None else '-':>8s}  {w['level']}")
            worst = max(rows, key=lambda w: w["s_max"] / w["D"])
            first = results[worst["name"]]
            print(f"\nDeepest predicted scour at {worst['name']} (depth {worst['depth']:.1f} m). Unprotected verdict: {first['level'].upper()}. {first['why']}")
            print(f"Run with --turbine {worst['name']} for the full assessment and plot.")
            if not args.no_plots:
                os.makedirs(args.out, exist_ok=True)
                pile = Pile(D=tb["D_pile"], h=worst["depth"], sediment=sed)
                fname = os.path.join(args.out, "scour_farm_%s_%s.png" % (args.farm, worst["name"]))
                path = plot(first, rec, pile, fname, name="%s %s" % (cfg["name"], worst["name"]), tp_coef=args.tp_coef)
                print(f"  plot: {os.path.relpath(path)}")
        out = rows or [summary_row(k, Pile(D=tb["D_pile"], h=v["pile"]["h"], sediment=sed), v) for k, v in results.items()]
    else:
        if args.D is None or args.h is None or args.d50 is None:
            ap.error("give --farm, or --D, --h and --d50 (with --csv or the synthetic-record options)")
        if args.allowance is None:
            args.allowance = 0.5
        pile = Pile(D=args.D, h=args.h, sediment=Sediment(args.d50 * 1e-3))
        rec = load_csv(args.csv) if args.csv else synthetic(args.start, args.years or 5.0, hs_median=args.hs_median,
                                                           tp_coef=args.tp_coef, u_spring=args.u_spring, surge_per_m=args.surge, seed=args.seed)
        r = run_site("site", pile, rec, args)
        out = [summary_row("site", pile, r)]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda o: o.isoformat() if isinstance(o, datetime) else float(o) if isinstance(o, np.floating) else str(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
